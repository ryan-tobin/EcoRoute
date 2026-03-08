import gymnasium as gym 
from gymnasium import spaces 
import numpy as np 
import networkx as nx 
import logging 

logger = logging.getLogger(__name__)

class EcoRouteEnv(gym.Env):
    """
    Custom Gymnasium Env for routing
    Features Continous Spatial Embeddings for state observation and Dense Reward Shaping
    """
    metadata = {"render_modes": ["console"]}

    def __init__(self, graph: nx.MultiDiGraph, render_mode: str = None):
        super().__init__()
        self.graph = graph 
        self.render_mode = render_mode 

        self.nodes = list(self.graph.nodes())
        self.num_nodes = len(self.nodes)
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        self.idx_to_node = {i: node for i, node in enumerate(self.nodes)}

        # Action Space: Max outgoing edges
        out_degrees = [ self.graph.out_degree(n) for n in self.nodes]
        self.max_actions = max(out_degrees) if out_degrees else 1
        self.action_space = spaces.Discrete(self.max_actions)

        # Calculate grid boundaries for Min-Max Normalization
        self.min_x = min(data['x'] for _, data in self.graph.nodes(data=True))
        self.max_x = max(data['x'] for _, data in self.graph.nodes(data=True))
        self.min_y = min(data['y'] for _, data in self.graph.nodes(data=True))
        self.max_y = max(data['y'] for _, data in self.graph.nodes(data=True))

        # Obs vector
        self.observation_space = spaces.Box(
            low=-2.0,
            high=2.0,
            shape=(7,),
            dtype=np.float32
        )

        self.current_node_idx = None 
        self.destination_node_idx = None 
        self.path_taken = []
        self.total_carbon = 0.0

    def _get_obs(self) -> np.ndarray:
        """Translates current and destination nodes into a normalized spatial vector"""
        curr_node = self.idx_to_node[self.current_node_idx]
        dest_node = self.idx_to_node[self.destination_node_idx]

        c_x = self.graph.nodes[curr_node]['x']
        c_y = self.graph.nodes[curr_node]['y']
        d_x = self.graph.nodes[dest_node]['x']
        d_y = self.graph.nodes[dest_node]['y']

        # Normalize coords to [0,1] bounds
        norm_cx = (c_x - self.min_x) / (self.max_x - self.min_x + 1e-6)
        norm_cy = (c_y - self.min_y) / (self.max_y - self.min_y + 1e-6)
        norm_dx = (d_x - self.min_x) / (self.max_x - self.min_x + 1e-6)
        norm_dy = (d_y - self.min_y) / (self.max_y - self.min_y + 1e-6)

        # Calculate heading (delta) and distance
        delta_x = norm_dx - norm_cx 
        delta_y = norm_dy - norm_cy 
        dist = (delta_x**2 + delta_y**2)**0.5

        return np.array([norm_cx, norm_cy, norm_dx, norm_dy, delta_x,delta_y, dist],dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_node_idx = self.np_random.integers(0, self.num_nodes)
        self.destination_node_idx = self.np_random.integers(0, self.num_nodes)

        while self.current_node_idx == self.destination_node_idx:
            self.destination_node_idx= self.np_random.integers(0, self.num_nodes)

        self.path_taken = [self.current_node_idx]
        self.total_carbon = 0.0

        observation = self._get_obs()
        info = {'start_node': self.current_node_idx, "goal_node": self.destination_node_idx}

        return observation, info
    
    def step(self, action: int):
        current_node_id = self.idx_to_node[self.current_node_idx]
        goal_node_id = self.idx_to_node[self.destination_node_idx]
        
        # Distance BEFORE move
        curr_x = self.graph.nodes[current_node_id]['x']
        curr_y = self.graph.nodes[current_node_id]['y']
        goal_x = self.graph.nodes[goal_node_id]['x']
        goal_y = self.graph.nodes[goal_node_id]['y']
        dist_before = ((curr_x - goal_x)**2 + (curr_y - goal_y)**2)**0.5
        
        neighbors = list(self.graph.successors(current_node_id))
        
        if len(neighbors) == 0:
            # Edge case: Spawned on a dead end
            return self._get_obs(), -10.0, True, False, {"reason": "dead_end"}
            
        # Mathematically force the action to be a valid exit
        safe_action = action % len(neighbors)
        
        # Valid Action Execution
        next_node_id = neighbors[safe_action]
        next_node_idx = self.node_to_idx[next_node_id]
        
        edge_data = self.graph.get_edge_data(current_node_id, next_node_id)
        edge_key = list(edge_data.keys())[0]
        actual_edge = edge_data[edge_key]
        
        carbon_cost = float(actual_edge.get('carbon_cost', 10.0))
        self.total_carbon += carbon_cost
        
        # Distance AFTER move
        next_x = self.graph.nodes[next_node_id]['x']
        next_y = self.graph.nodes[next_node_id]['y']
        dist_after = ((next_x - goal_x)**2 + (next_y - goal_y)**2)**0.5
        
        # Dense Reward Shaping
        distance_reward = (dist_before - dist_after) * 10000 
        reward = distance_reward - (carbon_cost * 0.1)
        
        self.current_node_idx = next_node_idx
        self.path_taken.append(self.current_node_idx)
        
        terminated = False
        truncated = False
        info = {"carbon_step": carbon_cost, "current_total_carbon": self.total_carbon}
        
        # Termination Conditions
        if self.current_node_idx == self.destination_node_idx:
            terminated = True
            reward += 1000.0  
            info["reason"] = "goal_reached"
            
        elif len(self.path_taken) > self.num_nodes // 2: 
            truncated = True
            reward -= 200.0
            info["reason"] = "max_steps_exceeded"
            
        return self._get_obs(), reward, terminated, truncated, info


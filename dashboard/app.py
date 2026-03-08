import streamlit as st 
import folium 
from streamlit_folium import st_folium
import networkx as nx 
import osmnx as ox 
from stable_baselines3 import PPO 
import numpy as np
from pathlib import Path 
import sys 
import os 

# Dynamically add the root directory to Python's path so it can find 'src'
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.append(root_path)

from src.engine.environment import EcoRouteEnv

# - UI Config -
st.set_page_config(layout='wide', page_title='EcoRoute Optimizer', page_icon="🌱")

# - Caching Data & Models
@st.cache_resource
def load_infrastructure():
    graph_path = Path(root_path) / "data/processed/center_city_philadelphia_pennsylvania_usa_network.graphml"
    model_path = Path(root_path) / "models/ppo_ecoroute.zip"

    G = ox.load_graphml(graph_path)
    env = EcoRouteEnv(G)
    model = PPO.load(model_path, env=env)

    return G, env, model 

G, env, model = load_infrastructure()

# - App Layout -
st.title("EcoRoute: RL-Powered Logistics")
st.markdown("**Demonstration:** Comparing traditional shortest-distance routing vs.Carbon-Optimized AI")


col1, col2 = st.columns([1,3])

with col1:
    st.subheader("Mission Control")
    st.markdown("Generate a random delivery request and let the PPO agent attempt to route it.")

    if st.button("Dispatch Vehicle", use_container_width=True):
        obs, info = env.reset()
        start_node_idx = info['start_node']
        goal_node_idx = info['goal_node']

        start_node_id = env.idx_to_node[start_node_idx]
        goal_node_id= env.idx_to_node[goal_node_idx]

        try:
            baseline_path = nx.shortest_path(G, start_node_id, goal_node_id, weight='length')
        except nx.NetworkXNoPath:
            baseline_path = []

        rl_path = [start_node_id]
        done = False
        current_obs = obs 
        steps = 0

        while not done and steps < 500:
            action, _states = model.predict(current_obs, deterministic=True)
            current_obs, reward, terminated, truncated, step_info = env.step(action)
            
            next_node_idx = env.current_node_idx
            next_node_id = env.idx_to_node[next_node_idx]

            if rl_path.count(next_node_id) > 3:
                break

            rl_path.append(next_node_id)
            
            if terminated or truncated:
                done = True
            steps += 1

        st.session_state['baseline'] = baseline_path
        st.session_state['rl_path'] = rl_path 

    if 'rl_path' in st.session_state:
        st.divider()
        st.metric("AI Route Steps Taken", len(st.session_state['rl_path']))
        st.caption("Note: Since the model only trained for 150k steps, the AI path migh wander or terminate early to avoid penalties")

with col2:
    if 'baseline' in st.session_state:
        start_id = st.session_state['baseline'][0] if st.session_state['baseline'] else st.session_state['rl_path'][0]
        start_lat = G.nodes[start_id]['y']
        start_lon = G.nodes[start_id]['x']

        m = folium.Map(location=[start_lat, start_lon], zoom_start=15, titles="CartoDB dark_matter")

        if len(st.session_state['baseline']) > 1:
            baseline_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in st.session_state['baseline']]
            folium.PolyLine(baseline_coords, color="#ff4b4b", weight=4,opacity=0.7, tooltip="Shortest Distance").add_to(m)

        if len(st.session_state['rl_path']) > 1:
            rl_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in st.session_state['rl_path']]
            folium.PolyLine(rl_coords, color="#00ff00", weight=5,opacity=0.9, tooltip="AI Carbon Route").add_to(m)

            folium.Marker(rl_coords[0],popup="Start",icon=folium.Icon(color="green", icon="play")).add_to(m)
            folium.Marker(rl_coords[-1], popup="Agent Stopped", icon=folium.Icon(color="red", icon="stop")).add_to(m)

        st_folium(m, width=900, height=600, returned_objects=[])
    else:
        st.info("Click **Dispatch Vehicle** to visualize the routing engine.")


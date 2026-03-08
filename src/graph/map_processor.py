import logging 
import osmnx as ox 
import networkx as nx 
from pathlib import Path 
from typing import Optional, Dict, Any 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MapProcessor:
    """
    Downloads, processes, and manages geospatial road network graphs.
    Transforms raw OpenStreetmap data into weighted Reinforcement Learning environment.
    """

    # Default sped limits (in km/h) to impute missing data based on road type
    DEFAULT_SPEEDS: Dict[str, float] = {
        'motorway': 100.0,
        'trunk': 80.0,
        'primary': 60.0,
        'secondary': 50.0,
        'tertiary': 40.0,
        'residential': 30.0,
        'living_street': 20.0,
        'unclassified': 30.0
    }

    def __init__(self, location_query: str, data_dir: str = 'data/processed'):
        self.location_query = location_query
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        safe_name = location_query.lower().replace(", ", "_").replace(" ", "_")
        self.graph_path = self.data_dir / f"{safe_name}_network.graphml"
        self.graph: Optional[nx.MultiDiGraph] = None 

    def fetch_or_load_graph(self) -> nx.MultiDiGraph:
        """Loads the graph from disk if cached, otherwise downloads from OSM."""
        if self.graph_path.exists():
            logger.info(f"Loading cached graph from {self.graph_path}")
            self.graph = ox.load_graphml(self.graph_path)
        else:
            logger.info(f"Downloading road network for: {self.location_query}. This may take a minute...")
            self.graph = ox.graph_from_place(self.location_query, network_type='drive')
            logger.info(f"Download complete. Nodes: {len(self.graph.nodes)}, Edges: {len(self.graph.edges)}")

            self._process_graph()
            self._save_graph()
        
        return self.graph 
    
    def _process_graph(self) -> None:
        """Runs the data cleaning and feature engineering pipeline."""
        if self.graph is None:
            raise ValueError("Graph has not been loaded")
        
        logger.info("Imputing missing speed limits...")
        self.graph = ox.add_edge_speeds(self.graph, fallback=self.DEFAULT_SPEEDS)

        logger.info("Calculating travel times...")
        self.graph = ox.add_edge_travel_times(self.graph)

        logger.info("Calculating carbon emission weights...")
        self._calculate_carbon_weights()

    def _calculate_carbon_weights(self) -> None:
        """
        Calculates a heuristic carbon cost for each edge.
        Base logic: CO2 = Distance (meters) * Emission Factor
        Emission Factor varies by speed (stop and go vs optimal vs high drag)
        """

        for u, v, key, data in self.graph.edges(keys=True, data=True):
            length_meters = float(data.get('length', 1.0))
            speed_kph = float(data.get('speed_kph', 30.0))

            # Base emission rate: ~0.15 grams of CO2 per meter for a delivery van
            base_emission_per_meter = 0.15

            # Non-linear emission multiplier based on speed efficiency
            if speed_kph < 35.0:
                multiplier = 1.4
            elif 35.0 <= speed_kph <= 80.0:
                multiplier = 1.0
            else:
                multiplier = 1.2
            
            carbon_cost = length_meters * base_emission_per_meter * multiplier

            # Add as a new edge attribute for the RL agent to optimize against
            self.graph[u][v][key]['carbon_cost'] = round(carbon_cost, 2)

    def _save_graph(self) -> None:
        """Serializes the processed graph to disk"""
        if self.graph is None:
            raise ValueError("No graph to save")
        
        logger.info(f"Saving processed graph to {self.graph_path}")
        ox.save_graphml(self.graph, self.graph_path)
        logger.info("Save complete")

if __name__ == "__main__":
    # Test the pipeline execution
    # We use a localized query to ensure the graph isn't too large for initial RAM limits
    # The 'Center City' constraint keeps the node count manageable for rapid testing.
    processor = MapProcessor(location_query="Center City, Philadelphia, Pennsylvania, USA")
    G = processor.fetch_or_load_graph()
    
    # Verify the engineering worked by inspecting a random edge
    edges = list(G.edges(data=True))
    if edges:
        sample_edge = edges[0][2]
        logger.info(f"Pipeline verification successful. Sample edge attributes: "
                    f"Length: {sample_edge.get('length')}m, "
                    f"Speed: {sample_edge.get('speed_kph')}kph, "
                    f"Carbon Cost: {sample_edge.get('carbon_cost')}g")
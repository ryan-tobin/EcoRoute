import logging 
from pathlib import Path 
import osmnx as ox 

# from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Import our custom environment
# Note: This is run from the root dir, so 'src.engine.environment' is the path.
from src.engine.environment import EcoRouteEnv

# Config logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_agent():
    # 1. Load data
    graph_path = Path("data/processed/center_city_philadelphia_pennsylvania_usa_network.graphml")

    if not graph_path.exists():
        logger.error(f"Graph not found at {graph_path}. Please run map_processor.py first")
        return 
    
    logger.info("Loading graph into memory...")
    G = ox.load_graphml(graph_path)

    # 2. Initialize the environment
    logger.info("Initializing EcoRoute Environment")
    env = EcoRouteEnv(G)

    # 3. Validate the Environment Architecture
    logger.info("Running strict API compliance check on our custom environment...")
    check_env(env, warn = True)
    logger.info("Environment passed compliance check!")

    # 4. Instantiate the NN
    logger.info("Instantiating PPO Agent...")
    # model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, tensorboard_log="./tensorboard_logs/")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard_logs/")

    # 5. Train the model
    logger.info("Starting training loop (500,000 timesteps)...")
    model.learn(total_timesteps=500000)

    # 6. Save the weights
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "RecurrentPPO_ecoroute"
    model.save(model_path)
    logger.info(f"Training complete. Model weights saved to {model_path}.zip")

if __name__ == "__main__":
    train_agent()

import sys
import logging
from . import data_pipeline, model_engine, prediction_system

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    if len(sys.argv) > 1:
        command = sys.argv[1]
    else:
        command = 'all'

    if command == 'train' or command == 'all':
        logger.info("Starting Training Pipeline...")
        logger.info("Step 1: Data Preparation")
        df = data_pipeline.run_pipeline()
        
        logger.info("Step 2: Model Training")
        engine = model_engine.HotspotModel()
        engine.train(df)
        
    if command == 'predict' or command == 'all':
        logger.info("Starting Prediction Pipeline...")
        pred_sys = prediction_system.PredictionSystem()
        try:
            pred_sys.load_model()
            pred_sys.generate_predictions()
        except FileNotFoundError:
            logger.error("Model not found. Run 'train' first.")

if __name__ == "__main__":
    main()

from src.mlproject.config.configuration import ConfigurationManager
from src.mlproject.components.model_trainer import ModelTrainer
from src.mlproject.logging import logger
from pathlib import Path


STAGE_NAME = "Model Trainer Stage"

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config()
            model_trainer_config = ModelTrainer(config=model_trainer_config)
            model_trainer_config.train()
        except Exception as e:
            raise e


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} Started <<<<<<")
        obj = ModelTrainerTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} Completed <<<<<<\n\nx=================x")
    except Exception as e:
        raise e
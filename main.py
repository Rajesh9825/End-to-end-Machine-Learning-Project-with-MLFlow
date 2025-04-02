from src.mlproject.logging import logger
from src.mlproject.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline



STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>>>> Stage {STAGE_NAME} Started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> Stage {STAGE_NAME} Completed <<<<<<\n\nx=================x")
except Exception as e:
    raise e
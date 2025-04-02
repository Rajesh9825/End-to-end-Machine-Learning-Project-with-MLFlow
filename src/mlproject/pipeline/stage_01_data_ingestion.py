from src.mlproject.config.configuration import ConfigurationManager
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.logging import logger

STAGE_NAME = "Data Ingestion Stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try: 
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingetsion = DataIngestion(config=data_ingestion_config)
            data_ingetsion.download_file()
            data_ingetsion.extract_zip_file()
        except Exception as e:
            raise e
        


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} Started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} Completed <<<<<<\n\nx=================x")
    except Exception as e:
        raise e
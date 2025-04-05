import os
import pandas as pd
from src.mlproject.utils.common import save_json
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from mlflow.models.signature import infer_signature
from src.mlproject.entity.config_entity import ModelEvaluationConfig
from src.mlproject.utils.common import save_json
from pathlib import Path

os.environ["export MLFLOW_TRACKING_URI"]="https://dagshub.com/rajuu9825/End-to-end-Machine-Learning-Project-with-MLFlow.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="rajuu9825"
os.environ["MLFLOW_TRACKING_PASSWORD"]="e9d3dc7e0e33d09d6992ea6a541706f25a7d1007"

class ModelEvaluation:
    def __init__(self,config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self,actual,pred):
        rmse = np.sqrt(mean_squared_error(actual,pred))
        mae = mean_absolute_error(actual,pred)
        r2 = r2_score(actual,pred)
        
        return rmse,mae,r2
    


    def log_into_mlflow(self):

        # Load test data and model
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        # Set MLflow Tracking URI (DagsHub or local)
        mlflow.set_tracking_uri(self.config.mlflow_uri)  # e.g., https://dagshub.com/your_username/your_repo.mlflow

        # Optional: set username/token as env vars if needed
        # os.environ["MLFLOW_TRACKING_USERNAME"] = "your_username"
        # os.environ["MLFLOW_TRACKING_PASSWORD"] = "your_token"

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Set or create experiment
        mlflow.set_experiment("wine-quality")

        with mlflow.start_run():

            # Predictions and signature
            predicted_qualities = model.predict(test_x)
            signature = infer_signature(test_x, predicted_qualities)

            # Evaluation
            rmse, mae, r2 = self.eval_metrics(test_y, predicted_qualities)

            # Save metrics locally (optional)
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            # Log params and metrics
            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            # Log model and register it
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name="WineQualityModel",  # This name will show in Model Registry
                    signature=signature
                )
            else:
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    signature=signature
                )

            print("MLflow logging complete")



                



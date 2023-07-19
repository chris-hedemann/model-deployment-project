import os
from dotenv import load_dotenv, set_key
import pandas as pd

import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline

import optuna
import optuna.visualization as ov
from optuna.integration.mlflow import MLflowCallback

from src.preprocess import read_dataframe, preprocess

def tune_model(X_train: list[dict], y_train: pd.Series, 
               MLFLOW_TRACKING_URI: str):
    """Tune model using optima to find best hyperparameters.
    Log each experiment run in MLflow

    Args:
        X_train (list[dict]): Training features
        y_train (pd.Series): Target
        MLFLOW_TRACKING_URI (str): MLFlow uri

    Returns:
        optima study object: final results of the study
    """

    mlflc = MLflowCallback(
        tracking_uri=MLFLOW_TRACKING_URI,
        metric_name="rmse"
    )

    dv = DictVectorizer()
    X_train = dv.fit_transform(X_train.copy())

    # Define the objective function for optimization
    @mlflc.track_in_mlflow()
    def objective(trial):
        # Define the search space for hyperparameters
        max_samples = trial.suggest_float('max_samples', 0.2, 0.9)
        # max_depth = trial.suggest_int('max_depth', 3, 6)
        criterion = trial.suggest_categorical('criterion',
                                ["squared_error",
                                "friedman_mse", 
                                "poisson"]
                                )
        # n_estimators = trial.suggest_int('n_estimators', 5, 20)
        ccp_alpha = trial.suggest_float('ccp_alpha', 0, 2)

        # Create the logistic regression model with the suggested hyperparameters
        rfreg = RandomForestRegressor(
            max_depth=4,
            n_estimators=2,
            max_samples=max_samples, 
            criterion=criterion,
            ccp_alpha=ccp_alpha
            )
        
        # Perform cross-validation
        score = cross_val_score(
            rfreg, 
            X_train, y_train,
            cv=5,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1)

        return score.mean()

    sampler = optuna.samplers.RandomSampler(seed=42)
    study = optuna.create_study(
        study_name="yellow_taxi_optuna_", 
        direction='maximize',
        sampler=sampler
        )

    # Optimize the objective function
    study.optimize(objective, n_trials=2, callbacks=[mlflc])

    # Print the best hyperparameters and best score
    print("Best Hyperparameters:", study.best_params)
    print("Best Score:", study.best_value)

    return study


def save_parameter_importance(study):
    """Saves parameter importance graph as html string.

    Args:
        study (optuna study object): A completed study
        from optuna
    """
    fig = ov.plot_param_importances(study)
    fig.write_html("./images/optima_output.html", 
                   full_html=False, 
                   include_plotlyjs='cdn')
    

def save_best_model(X_train: list[dict], 
        X_test: list[dict], 
        y_train: pd.Series,
        y_test: pd.Series, 
        tags: dict,
        MLFLOW_TRACKING_URI: str,
        EXPERIMENT_NAME: str,
        model_name="yellow_taxi_best_model_",
        replace_run_id=False,
        **model_kwargs) -> str:
    """Accepts best model parameters 

    Args:
        X_train (list[dict]): Training features
        X_test (list[dict]): Test features
        y_train (pd.Series): Training target
        y_test (pd.Series): Test target
        tags (dict): tags for the MLFlow experiment
        MLFLOW_TRACKING_URI (str): URI for MLflow
        EXPERIMENT_NAME (str): name of experiment
        model_name (str, optional): Defaults to "yellow_taxi_best_model".
        replace_run_id (bool): Replace the run ID in the .env file? 
Defaults to False.
        **model_kwargs (dict): the best model hyperparameters from the tuning step

    Returns:
        str: the Run ID
    """
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run: 
        mlflow.set_tags(tags)

        pipeline = make_pipeline(
        DictVectorizer(),
        RandomForestRegressor(
            max_depth=4,
            n_estimators=2,
            n_jobs=-1,
            **model_kwargs
            )
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(
            pipeline, "model",
            registered_model_name=model_name
            )

    RUN_ID = run.info.run_id

    if replace_run_id:
        set_key(".env", "RUN_ID", RUN_ID, quote_mode='never')

    return RUN_ID
    

# Adjust with click module to accept user input 
# at later stage. For now just work with hardcoded tags 
# and training parameters
def run():
    color = "yellow"
    year = 2021
    month = 1
    features = ["PULocationID", "DOLocationID", "trip_distance"]
    target = 'duration'
    tags = {
    "model": "random forest regressor",
    "developer": "chris-hedemann",
    "dataset": f"{color}-taxi",
    "year": year,
    "month": month,
    "features": features,
    "target": target
    }

    load_dotenv()
    MLFLOW_TRACKING_URI=os.getenv("MLFLOW_TRACKING_URI")
    EXPERIMENT_NAME=os.getenv("EXPERIMENT_NAME")
    SA_KEY=os.getenv("GOOGLE_SA_KEY")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SA_KEY

    print("Read and preprocess data\n\n")
    df = read_dataframe(color, year, month)
    X_train, X_test, y_train, y_test = preprocess(df)

    study = tune_model(X_train, y_train, 
                       MLFLOW_TRACKING_URI)
    
    # print("Saving parameter importance\n\n")
    # save_parameter_importance(study)
    
    print("Logging model and new run id\n\n")
    save_best_model(X_train, X_test, y_train, y_test, 
                    tags,
                    MLFLOW_TRACKING_URI,
                    EXPERIMENT_NAME,
                    **study.best_params)

    print("Done")


if __name__ == "__main__":
    run()

import os
import pandas as pd
import click
import mlflow
from src.preprocess import preprocess, generate_uuids

def load_model(run_id, mlflow_uri):
    mlflow.set_tracking_uri(mlflow_uri)
    logged_model = f'runs:/{run_id}/model'
    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    return loaded_model

def read_parquet_file(filename):
    df = pd.read_parquet(filename)
    df['trip_duration_minutes'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.trip_duration_minutes = df.trip_duration_minutes.dt.total_seconds() / 60
    df = df[(df.trip_duration_minutes >= 1) & (df.trip_duration_minutes <= 60)]
    df['ride_id'] = generate_uuids(len(df))
    return df

def save_results(df, y_pred, run_id, output_filename):
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['tpep_pickup_datetime'] = df['tpep_pickup_datetime']
    df_result['PULocationID'] = df['PULocationID']
    df_result['DOLocationID'] = df['DOLocationID']
    df_result['actual_duration'] = df['trip_duration_minutes']
    df_result['predicted_duration'] = y_pred
    df_result['diff'] = df_result['actual_duration'] - df_result['predicted_duration']
    df_result['model_version'] = run_id
    df_result.to_parquet(output_filename, index=False)


def apply_model(filename, run_id, output_filename, mlflow_uri):
    df = read_parquet_file(filename)
    dicts = preprocess(df)
    
    loaded_model = load_model(run_id, mlflow_uri)
    y_pred = loaded_model.predict(dicts)
    
    save_results(df, y_pred, run_id, output_filename)
    
@click.command()
@click.option("--filename", help="Path to the input parquet file")
@click.option("--run_id", help="MLflow run ID")
@click.option("--output_filename", help="Path to the output parquet file")
@click.option("--mlflow_uri", help="MLflow tracking URI")
@click.option("--google_sa_key", help="Path to the Google SA key")
def run(filename, run_id, output_filename, mlflow_uri, google_sa_key):
    filename = filename
    output_filename = output_filename
    run_id = run_id
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_sa_key
    apply_model(filename,
                run_id,
                output_filename,
                mlflow_uri)
    
    

if __name__ == "__main__":
    run()
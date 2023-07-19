from fastapi import FastAPI
from data_model import TaxiRide, TaxiRidePrediction
from predict import predict
import pandas as pd


app = FastAPI()

model_name = "yellow-taxi-trip-duration"

@app.get("/")
def index():
    return {"message": "NYC Yellow Taxi Ride Duration Prediction"}

@app.post("/predict", response_model=TaxiRidePrediction)
def predict_duration(data: TaxiRide):
    prediction = predict(model_name, data)
    return TaxiRidePrediction(**data.dict(), predicted_duration=prediction)


@app.post("/predict_batch", response_model=list[TaxiRidePrediction])
def predict_batch(data: list[TaxiRide] ):
    predictions = []
    for ride in data:
        prediction = predict(model_name, ride)
        predictions.append(
            TaxiRidePrediction(
            **ride.model_dump(),
            predicted_duration=prediction
            )
        )
    return predictions

@app.post("/predict_bq", response_model=list[TaxiRidePrediction])
def predict_bg(data: list[TaxiRide], 
               table_name='mlflow_stream.yellow_taxi_predictions_api',
               if_exists='append'):

    predictions = predict_batch(data)
    df = pd.DataFrame.from_dict(predictions, orient='columns')

    try:
        df.to_gbq(
            destination_table=table_name,
            chunksize=10000,
            if_exists=if_exists,
            )

    except ConnectionError:
        print("The connection to BigQuery failed")
        return predictions
        
    return predictions


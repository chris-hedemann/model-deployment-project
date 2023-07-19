from pydantic import BaseModel, PositiveFloat

class TaxiRide(BaseModel):
    ride_id: str
    PULocationID: int
    DOLocationID: int
    trip_distance: float

class TaxiRidePrediction(TaxiRide):
    predicted_duration: PositiveFloat

import uuid
from src.load import load_data
from sklearn.model_selection import train_test_split


def generate_uuids(n):
    ride_ids = []
    for i in range(n):
        ride_ids.append(str(uuid.uuid4()))
    return ride_ids

def read_dataframe(color: str, year: int=2021, month: int=1):
    df = load_data(color, year, month)
    df['trip_duration_minutes'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.trip_duration_minutes = df.trip_duration_minutes.dt.total_seconds() / 60
    df = df[(df.trip_duration_minutes >= 1) & (df.trip_duration_minutes <= 60)]
    df['ride_id'] = generate_uuids(len(df))
    return df

def preprocess(df):
    df = df.copy()
    categorical_features = ["PULocationID", "DOLocationID"]
    df[categorical_features] = df[categorical_features].astype(str)
    df['trip_route'] = df["PULocationID"] + "_" + df["DOLocationID"]
    dicts = df[['trip_route', 'trip_distance']].to_dict(orient='records')
    
    return dicts
    
def preprocess_train(df, random_state=42, test_size=0.2):
    df = df.copy()
    categorical_features = ["PULocationID", "DOLocationID"]
    df[categorical_features] = df[categorical_features].astype(str)
    df['trip_route'] = df["PULocationID"] + "_" + df["DOLocationID"]
    df = df[['trip_route', 'trip_distance', 'trip_duration_minutes']]

    y=df["trip_duration_minutes"]
    X=df.drop(columns=["trip_duration_minutes"])

    X_train, X_test, y_train, y_test = \
    train_test_split(X, y, random_state=random_state, 
                     test_size=test_size)
    X_train = X_train.to_dict(orient="records")
    X_test = X_test.to_dict(orient="records")

    return X_train, X_test, y_train, y_test
    



    
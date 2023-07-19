import pandas as pd
import os
import click

def load_data(color: str, year: int, month: int):
    """Load data-set from local directory"""
    # Download the data
    year, month = int(year), int(month)
    file_name = f"{color}_tripdata_{year}-{month:02d}.parquet"
    if not os.path.exists(f"./data/{file_name}"):
        os.system(
            f"wget -P ./data https://d37ci6vzurychx.cloudfront.net/trip-data/{file_name}"
            )
    
    df = pd.read_parquet(f'./data/{file_name}')
    return df

@click.command()
@click.option("--color")
@click.option("--year")
@click.option("--month")
def run(color, year, month):
    year = int(year)
    month = int(month)
    load_data(color, year, month)

if __name__ == "__main__":
    run()
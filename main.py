from src.etl import main as run_etl
from src.Train.train import train


def run_pipeline():
    print("Starting ETL pipeline...")
    run_etl()

    print("Starting training pipeline...")
    train()

    print("Pipeline finished successfully.")


if __name__ == "__main__":
    run_pipeline()
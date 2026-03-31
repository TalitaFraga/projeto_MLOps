from pathlib import Path
import pandas as pd

from src.Extract.extract import GenZExtract
from src.Transform.transform import GenZTransform
from src.Load.load import GenZLoad
from src.Train.train import train


def run_pipeline():
    print("Starting ETL pipeline...")

    extractor = GenZExtract()
    raw_file_path = extractor.extract()
    print(f"Extraction completed: {raw_file_path}")

    print("Loading raw data...")
    df_raw = pd.read_csv(raw_file_path)
    print("Raw data loaded successfully.")

    transformer = GenZTransform()
    df_transformed = transformer.transform(df_raw)
    print("Transformation completed.")

    loader = GenZLoad()
    processed_file_path = loader.load(df_transformed)
    print(f"Processed data saved at: {processed_file_path}")

    print("Starting training pipeline...")
    train()
    print("Training pipeline completed.")

    print("Pipeline finished successfully.")


if __name__ == "__main__":
    run_pipeline()
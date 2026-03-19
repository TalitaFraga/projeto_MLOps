from pathlib import Path
import pandas as pd


class GenZLoad:
    def __init__(self, output_path: str = "data/processed/processed_genz_mental_wellness_synthetic_dataset.csv"):
        self.project_root = Path(__file__).resolve().parents[2]
        self.output_file = self.project_root / output_path

    def load(self, df: pd.DataFrame) -> Path:
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_file, index=False)
        return self.output_file
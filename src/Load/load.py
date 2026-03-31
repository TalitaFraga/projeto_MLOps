from pathlib import Path
import os

import pandas as pd
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


class GenZLoad:
    def __init__(self, output_path: str | None = None):
        self.project_root = Path(__file__).resolve().parents[2]
        self.output_file = self.project_root / (
            output_path or os.getenv("PROCESSED_DATA_PATH")
        )

    def load(self, df: pd.DataFrame) -> Path:
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_file, index=False)
        return self.output_file
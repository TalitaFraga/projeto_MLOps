from pathlib import Path

import pandas as pd

from src.config import BASE_DIR, get_param


class GenZLoad:
    def __init__(self, output_path: str | None = None):
        self.project_root = BASE_DIR
        self.output_file = self.project_root / (
            output_path or get_param("paths", "processed_data_path")
        )

    def load(self, df: pd.DataFrame) -> Path:
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_file, index=False)
        return self.output_file
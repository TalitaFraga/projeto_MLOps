from pathlib import Path
import os
import pandas as pd
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


class GenZTransform:
    def __init__(self):
        categorical_cols = os.getenv("CATEGORICAL_COLS", "")
        drop_columns = os.getenv("DROP_COLUMNS", "")

        self.categorical_cols = [col.strip() for col in categorical_cols.split(",") if col.strip()]
        self.drop_columns = [col.strip() for col in drop_columns.split(",") if col.strip()]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        original_order = [col for col in df.columns if col not in self.drop_columns]

        existing_drop_columns = [col for col in self.drop_columns if col in df.columns]
        if existing_drop_columns:
            df = df.drop(columns=existing_drop_columns)

        categorical_cols_existing = [col for col in self.categorical_cols if col in df.columns]
        df = pd.get_dummies(df, columns=categorical_cols_existing, dtype=int)

        new_order = []
        for col in original_order:
            if col in categorical_cols_existing:
                encoded_cols = [c for c in df.columns if c.startswith(f"{col}_")]
                new_order.extend(encoded_cols)
            else:
                if col in df.columns:
                    new_order.append(col)

        df = df[new_order]

        return df
import pandas as pd

from src.config import get_param


class GenZTransform:
    def __init__(self):
        self.categorical_cols = get_param("transform", "categorical_cols", default=[])
        self.drop_columns = get_param("transform", "drop_columns", default=[])

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
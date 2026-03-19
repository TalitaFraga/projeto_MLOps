import pandas as pd


class GenZTransform:
    def __init__(self):
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        categorical_cols = [
            "Gender",
            "Student_Working_Status",
            "Content_Type_Preference"
        ]

        original_order = [col for col in df.columns if col != "Country"]

        if "Country" in df.columns:
            df = df.drop(columns=["Country"])

        df = pd.get_dummies(df, columns=categorical_cols, dtype=int)

        new_order = []
        for col in original_order:
            if col in categorical_cols:
                encoded_cols = [c for c in df.columns if c.startswith(f"{col}_")]
                new_order.extend(encoded_cols)
            else:
                new_order.append(col)

        df = df[new_order]

        return df
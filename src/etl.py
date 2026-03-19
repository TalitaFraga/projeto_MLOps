from pathlib import Path

from Extract.extract import GenZExtract
from Transform.transform import GenZTransform
from Load.load import GenZLoad

def main():

    extractor = GenZExtract("../data/raw/genz_mental_wellness_synthetic_dataset.csv")
    transformer = GenZTransform()
    loader = GenZLoad("../data/processed/gen_z.csv")

    df_raw = extractor.extract()
    df_processed = transformer.transform(df_raw)
    loader.load(df_processed)

if __name__ == "__main__":
    main()
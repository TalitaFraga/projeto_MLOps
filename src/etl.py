import pandas as pd

from Extract.extract import GenZExtract
from Transform.transform import GenZTransform
from Load.load import GenZLoad


def main():
    extractor = GenZExtract()
    #transformer = GenZTransform()
    #loader = GenZLoad("data/processed/gen_z.csv")

    caminho_csv = extractor.extract()
    #df_raw = pd.read_csv(caminho_csv)
    #df_processed = transformer.transform(df_raw)
    #loader.load(df_processed)

    print("Pipeline executado com sucesso.")


if __name__ == "__main__":
    main()
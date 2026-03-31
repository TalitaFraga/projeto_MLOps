import pandas as pd

from Extract.extract import GenZExtract
from Transform.transform import GenZTransform
from Load.load import GenZLoad


def main():
    extractor = GenZExtract()
    transformer = GenZTransform()
    loader = GenZLoad()

    caminho_csv = extractor.extract()
    df_raw = pd.read_csv(caminho_csv)
    df_processed = transformer.transform(df_raw)
    caminho_saida = loader.load(df_processed)

    print(f"Pipeline finalizada. Arquivo salvo em: {caminho_saida}")


if __name__ == "__main__":
    main()
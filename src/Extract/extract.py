from pathlib import Path
import shutil
import kagglehub
import os

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


class GenZExtract:
    def __init__(self, dataset_id: str | None = None):
        self.project_root = Path(__file__).resolve().parents[2]

        self.dataset_id = dataset_id or os.getenv("KAGGLE_DATASET_ID")
        self.raw_dir = self.project_root / os.getenv("RAW_DATA_DIR")
        self.raw_data_path = self.project_root / os.getenv("RAW_DATA_PATH")

    def extract(self) -> Path:
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        download_dir = Path(kagglehub.dataset_download(self.dataset_id))
        csv_files = list(download_dir.rglob("*.csv"))

        csv_origem = csv_files[0]
        csv_destino = self.raw_data_path

        shutil.copy(csv_origem, csv_destino)

        return csv_destino


if __name__ == "__main__":
    extractor = GenZExtract()
    caminho_arquivo = extractor.extract()
    print(f"Arquivo extraído para: {caminho_arquivo}")
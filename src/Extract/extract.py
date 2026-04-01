from pathlib import Path
import shutil
import kagglehub

from src.config import BASE_DIR, get_param


class GenZExtract:
    def __init__(
        self,
        dataset_id: str | None = None,
        raw_dir: str | None = None,
        raw_data_path: str | None = None,
    ):
        self.project_root = BASE_DIR

        self.dataset_id = dataset_id or get_param("dataset", "kaggle_dataset_id")
        self.raw_dir = self.project_root / (raw_dir or get_param("paths", "raw_data_dir"))
        self.raw_data_path = self.project_root / (
            raw_data_path or get_param("paths", "raw_data_path")
        )

    def extract(self) -> Path:
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        download_dir = Path(kagglehub.dataset_download(self.dataset_id))
        csv_files = list(download_dir.rglob("*.csv"))

        if not csv_files:
            raise FileNotFoundError("Nenhum arquivo CSV foi encontrado no dataset baixado.")

        csv_origem = csv_files[0]
        csv_destino = self.raw_data_path

        shutil.copy(csv_origem, csv_destino)

        return csv_destino


if __name__ == "__main__":
    extractor = GenZExtract()
    caminho_arquivo = extractor.extract()
    print(f"Arquivo extraído para: {caminho_arquivo}")
from pathlib import Path
import shutil
import os
import kagglehub

from src.config import BASE_DIR, get_param, get_env


class GenZExtract:
    def __init__(
        self,
        dataset_id: str | None = None,
        raw_dir: str | None = None,
        raw_data_path: str | None = None,
        kaggle_api_token: str | None = None,
    ):
        self.project_root = BASE_DIR

        self.dataset_id = dataset_id or get_param("dataset", "kaggle_dataset_id")
        self.raw_dir = self.project_root / (raw_dir or get_param("paths", "raw_data_dir"))
        self.raw_data_path = self.project_root / (
            raw_data_path or get_param("paths", "raw_data_path")
        )

        self.kaggle_api_token = kaggle_api_token or get_env("KAGGLE_API_TOKEN", required=True)

        if not self.dataset_id:
            raise ValueError("Dataset ID não informado em params.yaml.")

    def _configure_kaggle_credentials(self) -> None:
        os.environ["KAGGLE_API_TOKEN"] = self.kaggle_api_token

    def extract(self) -> Path:
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        self._configure_kaggle_credentials()

        try:
            download_dir = Path(kagglehub.dataset_download(self.dataset_id))
        except Exception as e:
            raise RuntimeError(
                "Falha ao baixar o dataset do Kaggle. "
                "Verifique KAGGLE_API_TOKEN e dataset.kaggle_dataset_id."
            ) from e

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
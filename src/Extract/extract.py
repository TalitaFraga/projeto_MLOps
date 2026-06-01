from pathlib import Path
import os
import shutil

import kagglehub

from src.config import BASE_DIR, get_param, get_env


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

        if not self.dataset_id:
            raise ValueError("Dataset ID não informado em params.yaml.")

    def _configure_kaggle_credentials(self) -> None:
        """
        O kagglehub utiliza as variáveis KAGGLE_USERNAME e KAGGLE_KEY,
        ou o arquivo ~/.kaggle/kaggle.json.

        No Docker, o caminho recomendado é preencher o arquivo .env com:
        KAGGLE_USERNAME=...
        KAGGLE_KEY=...
        """
        kaggle_username = get_env("KAGGLE_USERNAME", default="")
        kaggle_key = get_env("KAGGLE_KEY", default="")
        kaggle_json_path = Path.home() / ".kaggle" / "kaggle.json"

        if kaggle_username and kaggle_key:
            os.environ["KAGGLE_USERNAME"] = kaggle_username
            os.environ["KAGGLE_KEY"] = kaggle_key
            return

        if kaggle_json_path.exists():
            return

        raise ValueError(
            "Credenciais do Kaggle não encontradas. "
            "Defina KAGGLE_USERNAME e KAGGLE_KEY no arquivo .env "
            "ou configure ~/.kaggle/kaggle.json."
        )

    def extract(self) -> Path:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self._configure_kaggle_credentials()

        try:
            download_dir = Path(kagglehub.dataset_download(self.dataset_id))
        except Exception as exc:
            raise RuntimeError(
                "Falha ao baixar o dataset do Kaggle. "
                "Verifique KAGGLE_USERNAME, KAGGLE_KEY e dataset.kaggle_dataset_id."
            ) from exc

        csv_files = sorted(download_dir.rglob("*.csv"))

        if not csv_files:
            raise FileNotFoundError("Nenhum arquivo CSV foi encontrado no dataset baixado.")

        csv_origem = csv_files[0]
        csv_destino = self.raw_data_path
        csv_destino.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(csv_origem, csv_destino)

        print(f"Dataset baixado do Kaggle: {self.dataset_id}")
        print(f"Arquivo copiado para: {csv_destino}")

        return csv_destino


if __name__ == "__main__":
    extractor = GenZExtract()
    caminho_arquivo = extractor.extract()
    print(f"Arquivo extraído para: {caminho_arquivo}")
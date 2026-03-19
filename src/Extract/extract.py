from pathlib import Path
import shutil
import kagglehub


class GenZExtract:
    def __init__(
        self,
        dataset_id: str = "hammadansari7/gen-z-mental-wellness-and-digital-lifestyle-patterns"
    ):
        self.dataset_id = dataset_id
        self.project_root = Path(__file__).resolve().parents[2]
        self.raw_dir = self.project_root / "data" / "raw"

    def extract(self) -> Path:
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        download_dir = Path(kagglehub.dataset_download(self.dataset_id))

        csv_files = list(download_dir.rglob("*.csv"))

        csv_origem = csv_files[0]
        csv_destino = self.raw_dir / csv_origem.name

        shutil.copy(csv_origem, csv_destino)

        return csv_destino


if __name__ == "__main__":
    extractor = GenZExtract()
    extractor.extract()
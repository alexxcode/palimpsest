"""Download LEVIR-CD dataset from Google Drive."""
import zipfile
from pathlib import Path

import gdown

# Official LEVIR-CD dataset — justchenhao.github.io/LEVIR/
LEVIR_GDRIVE_ID = "1dLuzldMRmbBNKPpUkX8Z53hi6NHLrWim"

OUTPUT_DIR = Path("data/datasets")   # splits land as data/datasets/train|val|test/


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    archive_path = OUTPUT_DIR.parent / "levir_cd.zip"
    url = f"https://drive.google.com/uc?id={LEVIR_GDRIVE_ID}"

    print(f"Downloading LEVIR-CD to {archive_path} ...")
    gdown.download(url, str(archive_path), quiet=False)

    print("Extracting ...")
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(OUTPUT_DIR.parent)

    archive_path.unlink()
    print(f"Done. Dataset at {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

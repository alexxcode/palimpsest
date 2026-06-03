"""Download pre-trained ChangeFormer checkpoint from open-cd model zoo."""
# Checkpoint source: https://github.com/likyoo/open-cd (check configs/changeformer/)
# Verify the URL against the latest open-cd release before running.
from pathlib import Path

import gdown

CHECKPOINT_DIR = Path("model/checkpoints")

# ChangeFormer trained on LEVIR-CD — from open-cd model zoo
# TODO (Phase 4): confirm exact URL from open-cd model zoo table
CHECKPOINT_GDRIVE_ID = "PLACEHOLDER"
CHECKPOINT_NAME = "changeformer_levir.pth"


def main() -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    dest = CHECKPOINT_DIR / CHECKPOINT_NAME

    if dest.exists():
        print(f"Checkpoint already exists: {dest}")
        return

    if CHECKPOINT_GDRIVE_ID == "PLACEHOLDER":
        raise RuntimeError(
            "Set CHECKPOINT_GDRIVE_ID in this script before running. "
            "Find the link at: https://github.com/likyoo/open-cd"
        )

    url = f"https://drive.google.com/uc?id={CHECKPOINT_GDRIVE_ID}"
    print(f"Downloading ChangeFormer checkpoint to {dest} ...")
    gdown.download(url, str(dest), quiet=False)
    print("Done.")


if __name__ == "__main__":
    main()

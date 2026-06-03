"""
Palimpsest — ChangeFormer fine-tune on Kaggle (T4 GPU)
Fine-tune 3: epochs 251-325, pos_weight=2.5, peak_lr=5e-6

Paste this entire file as a single code cell in a new Kaggle notebook.
Enable GPU: Settings → Accelerator → GPU T4 x2

Dataset needed — add via the + Add Data button on the right panel:
  * LEVIR-CD: search "LEVIR-CD Change Detection" by antonioring
  * Checkpoint: upload changeformer_levir.pth as a Kaggle dataset first
    (or download from GCS in the cell below if you have a service account key)
"""

# ── 0. Packages ──────────────────────────────────────────────────────────────
import subprocess, sys, os

subprocess.run([
    sys.executable, '-m', 'pip', 'install', '-q',
    'timm', 'einops', 'pydantic-settings', 'scipy',
    'google-cloud-storage',
], check=True)
print("Packages ready.")

# ── 1. Paths ──────────────────────────────────────────────────────────────────
import os, pathlib

# Kaggle datasets are mounted at /kaggle/input/<dataset-slug>/
# Adjust these to match your dataset names:
LEVIR_ROOT = "/kaggle/input/levir-cd"          # LEVIR-CD dataset slug
CKPT_IN    = "/kaggle/input/palimpsest-checkpoint/changeformer_levir.pth"
WORKDIR    = "/kaggle/working/palimpsest"
CKPT_OUT   = f"{WORKDIR}/checkpoints/changeformer_levir.pth"

os.makedirs(f"{WORKDIR}/checkpoints", exist_ok=True)

# ── 2. Download project code from GCS ────────────────────────────────────────
# Option A: if you have a GCS service account key saved as a Kaggle secret
# (Kaggle sidebar → Add-ons → Secrets → add key named GCS_SA_KEY)
try:
    from kaggle_secrets import UserSecretsClient
    sa_key = UserSecretsClient().get_secret("GCS_SA_KEY")
    with open("/tmp/sa_key.json", "w") as f:
        f.write(sa_key)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/sa_key.json"
    subprocess.run([
        "gsutil", "cp",
        "gs://palimpsest-bucket/code/palimpsest_code.tar.gz", WORKDIR
    ], check=True)
    print("Code downloaded from GCS.")
except Exception as e:
    print(f"GCS auth skipped ({e}) — downloading code via pip install or manual upload.")

# Option B: if you uploaded the code tarball to a Kaggle dataset
CODE_TAR = "/kaggle/input/palimpsest-code/palimpsest_code.tar.gz"
if os.path.exists(CODE_TAR):
    subprocess.run(["tar", "xzf", CODE_TAR, "-C", WORKDIR], check=True)
    print("Code unpacked from Kaggle dataset.")

with open(f"{WORKDIR}/.env", "w") as f:
    f.write("GCP_PROJECT_ID=project-8c7ca821-aa7a-45ea-88b\n"
            "GCS_BUCKET_NAME=palimpsest-bucket\n"
            "EE_PROJECT=project-8c7ca821-aa7a-45ea-88b\n")

os.chdir(WORKDIR)
if WORKDIR not in sys.path:
    sys.path.insert(0, WORKDIR)

# ── 3. Copy checkpoint ────────────────────────────────────────────────────────
import shutil

if os.path.exists(CKPT_IN):
    shutil.copy(CKPT_IN, CKPT_OUT)
    print(f"Checkpoint copied ({os.path.getsize(CKPT_OUT)/1e6:.1f} MB)")
else:
    print(f"WARNING: checkpoint not found at {CKPT_IN}")
    print("Add your checkpoint as a Kaggle dataset or download from GCS.")

# ── 4. Verify GPU ─────────────────────────────────────────────────────────────
import torch
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
else:
    raise RuntimeError("No GPU available — enable in Settings → Accelerator")

# ── 5. Check LEVIR-CD dataset ─────────────────────────────────────────────────
from pathlib import Path
for split in ("train", "val", "test"):
    a_dir = Path(LEVIR_ROOT) / split / "A"
    if a_dir.exists():
        n = len(list(a_dir.glob("*.png")))
        print(f"  {split}: {n} pairs  ({a_dir})")
    else:
        # Try alternative Kaggle dataset layouts
        alt = Path("/kaggle/input") / "levir-cd-change-detection" / split / "A"
        if alt.exists():
            LEVIR_ROOT = str(alt.parent.parent)
            print(f"  Found at: {LEVIR_ROOT}")
            break
        print(f"  WARNING: {a_dir} not found — check your dataset slug")

# ── 6. Fine-tune 3: epochs 251–325 ───────────────────────────────────────────
cmd = [
    sys.executable, "model/train.py",
    "--epochs",       "325",
    "--batch-size",   "16",
    "--lr",           "5e-6",
    "--warmup",       "1",
    "--pos-weight",   "2.5",
    "--workers",      "4",          # Kaggle has more CPUs
    "--val-interval", "5",
    "--data-root",    LEVIR_ROOT,
    "--checkpoint",   CKPT_OUT,
    "--resume",
    # "--gcs-upload",  # uncomment if GCS_SA_KEY secret is configured
]

print("Fine-tune 3: epochs 251–325  pos_weight=2.5  peak_lr=5e-6\n")
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1, cwd=WORKDIR)
for line in proc.stdout:
    print(line, end="", flush=True)
proc.wait()
print(f"\nExited {proc.returncode}")

# ── 7. Threshold sweep on val set ─────────────────────────────────────────────
print("\n\n=== Threshold sweep ===")
proc = subprocess.Popen(
    [sys.executable, "model/evaluate.py",
     "--checkpoint", CKPT_OUT,
     "--data-root",  LEVIR_ROOT,
     "--split", "val",
     "--threshold-sweep"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, bufsize=1, cwd=WORKDIR,
)
for line in proc.stdout:
    print(line, end="", flush=True)
proc.wait()

# ── 8. Save checkpoint to Kaggle output ──────────────────────────────────────
# Kaggle output files persist at /kaggle/working/ — download via the Output tab.
shutil.copy(CKPT_OUT, "/kaggle/working/changeformer_levir_ft3.pth")
print(f"\nCheckpoint saved to /kaggle/working/changeformer_levir_ft3.pth")
print("Download it from the Output tab, then upload to GCS:")
print("  gsutil cp changeformer_levir_ft3.pth gs://palimpsest-bucket/checkpoints/")

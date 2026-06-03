#!/bin/bash
# =============================================================================
# Palimpsest — VM training setup script
# Run on a GCP Deep Learning VM (PyTorch 2.9 + CUDA 12.9)
#
# Usage:
#   bash vm_setup.sh [--epochs 100] [--batch-size 16]
# =============================================================================
set -euo pipefail

PROJECT_ID="project-8c7ca821-aa7a-45ea-88b"
BUCKET="palimpsest-bucket"
WORKDIR="/home/user/palimpsest"
EPOCHS="${1:-100}"
BATCH="${2:-16}"

echo "=============================="
echo " Palimpsest VM Training Setup "
echo "=============================="

# ---- 1. Install extra Python deps not in the base image ------------------
pip install --quiet timm einops pydantic-settings scipy Pillow \
    google-cloud-storage google-cloud-bigquery

# ---- 2. Download and unpack training code --------------------------------
mkdir -p "$WORKDIR" && cd "$WORKDIR"
gsutil cp "gs://$BUCKET/code/palimpsest_code.tar.gz" .
tar xzf palimpsest_code.tar.gz
rm palimpsest_code.tar.gz

# Create minimal .env (settings read from env vars, no GCS/BQ calls during training)
cat > .env <<EOF
GCP_PROJECT_ID=$PROJECT_ID
GCS_BUCKET_NAME=$BUCKET
EE_PROJECT=$PROJECT_ID
EOF

# ---- 3. Download LEVIR-CD dataset from GCS --------------------------------
echo "Downloading LEVIR-CD from GCS..."
mkdir -p data/datasets
gsutil -m cp -r "gs://$BUCKET/datasets/*" data/datasets/
echo "Dataset ready: $(find data/datasets -name '*.png' | wc -l) PNG files"

# ---- 4. Run training ------------------------------------------------------
nvidia-smi  # sanity check GPU

echo ""
echo "Starting training: epochs=$EPOCHS  batch=$BATCH"
echo "Checkpoint will be saved to gs://$BUCKET/checkpoints/"
echo ""

python model/train.py \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH" \
    --lr 6e-5 \
    --workers 4 \
    --val-interval 5 \
    --gcs-upload \
    2>&1 | tee training_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "Training complete. Checkpoint in gs://$BUCKET/checkpoints/"

#!/usr/bin/env bash
set -euo pipefail

# Idempotent GCP infrastructure setup for Palimpsest.
# Run from inside the container: docker compose run --rm api bash infra/setup_gcp.sh

PROJECT="${GCP_PROJECT_ID}"
REGION="${GCP_REGION}"
BUCKET="${GCS_BUCKET_NAME}"
BQ_DATASET="${BQ_DATASET}"
BQ_TABLE="${BQ_TABLE}"
AR_REPO="${AR_REPOSITORY}"
SA_NAME="palimpsest-runner"
SA_EMAIL="${SA_NAME}@${PROJECT}.iam.gserviceaccount.com"

echo "=== Palimpsest GCP setup | project: ${PROJECT} ==="

# --- Cloud Storage ---
echo "--- Cloud Storage ---"
if ! gsutil ls "gs://${BUCKET}" &>/dev/null; then
  gsutil mb -p "${PROJECT}" -l "${REGION}" "gs://${BUCKET}"
  echo "Bucket created: gs://${BUCKET}"
else
  echo "Bucket already exists: gs://${BUCKET}"
fi
gsutil lifecycle set infra/gcs_lifecycle.json "gs://${BUCKET}"
echo "Lifecycle policy applied (raw/ deleted after 90 days)"

# --- BigQuery ---
echo "--- BigQuery ---"
if ! bq ls --project_id="${PROJECT}" "${BQ_DATASET}" &>/dev/null; then
  bq mk --project_id="${PROJECT}" --location="${REGION}" "${BQ_DATASET}"
  echo "Dataset created: ${BQ_DATASET}"
else
  echo "Dataset already exists: ${BQ_DATASET}"
fi

if ! bq show --project_id="${PROJECT}" "${BQ_DATASET}.${BQ_TABLE}" &>/dev/null; then
  bq mk --project_id="${PROJECT}" \
    --table "${BQ_DATASET}.${BQ_TABLE}" \
    infra/bq_schema.json
  echo "Table created: ${BQ_DATASET}.${BQ_TABLE}"
else
  echo "Table already exists: ${BQ_DATASET}.${BQ_TABLE}"
fi

# --- Artifact Registry ---
echo "--- Artifact Registry ---"
if ! gcloud artifacts repositories describe "${AR_REPO}" \
    --project="${PROJECT}" --location="${REGION}" &>/dev/null; then
  gcloud artifacts repositories create "${AR_REPO}" \
    --project="${PROJECT}" \
    --location="${REGION}" \
    --repository-format=docker
  echo "Repository created: ${AR_REPO}"
else
  echo "Repository already exists: ${AR_REPO}"
fi

# --- Service Account ---
echo "--- Service Account ---"
if ! gcloud iam service-accounts describe "${SA_EMAIL}" \
    --project="${PROJECT}" &>/dev/null; then
  gcloud iam service-accounts create "${SA_NAME}" \
    --project="${PROJECT}" \
    --display-name="Palimpsest Cloud Run runner"
  echo "Service account created: ${SA_EMAIL}"
else
  echo "Service account already exists: ${SA_EMAIL}"
fi

for ROLE in \
  "roles/storage.objectAdmin" \
  "roles/bigquery.dataEditor" \
  "roles/bigquery.jobUser" \
  "roles/earthengine.writer" \
  "roles/artifactregistry.reader"; do
  gcloud projects add-iam-policy-binding "${PROJECT}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="${ROLE}" \
    --condition=None \
    --quiet
  echo "Granted ${ROLE}"
done

echo "=== Setup complete ==="

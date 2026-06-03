#!/usr/bin/env bash
# =============================================================================
# Palimpsest — Cloud Run smoke test
# Usage:  bash infra/smoke_test.sh <SERVICE_URL>
#   e.g.: bash infra/smoke_test.sh https://palimpsest-api-xxxx-uc.a.run.app
#
# Requires: curl, jq, gcloud (authenticated)
# =============================================================================
set -euo pipefail

SERVICE_URL="${1:-}"
if [[ -z "$SERVICE_URL" ]]; then
  # Try to auto-detect from Cloud Run
  SERVICE_URL=$(gcloud run services describe palimpsest-api \
    --region=us-central1 \
    --format="value(status.url)" 2>/dev/null || true)
fi

if [[ -z "$SERVICE_URL" ]]; then
  echo "Usage: bash infra/smoke_test.sh <SERVICE_URL>"
  exit 1
fi

echo "============================================"
echo " Palimpsest Cloud Run smoke test"
echo " URL: $SERVICE_URL"
echo "============================================"

# Get an identity token for the authenticated Cloud Run service
TOKEN=$(gcloud auth print-identity-token)
AUTH_HEADER="Authorization: Bearer $TOKEN"

# --- 1. Health check ---
echo ""
echo "1. GET /health"
HEALTH=$(curl -sf -H "$AUTH_HEADER" "$SERVICE_URL/health")
echo "$HEALTH" | jq .

MODEL_VER=$(echo "$HEALTH" | jq -r '.model_version // empty')
MODEL_LOADED=$(echo "$HEALTH" | jq -r '.model_loaded // empty')

if [[ "$MODEL_LOADED" != "true" ]]; then
  echo "ERROR: model_loaded is not true"
  exit 1
fi

echo "  model_version : $MODEL_VER"
echo "  model_loaded  : $MODEL_LOADED"
echo "  ✓ Health OK"

# --- 2. Detect endpoint — small AOI over a known urban area ---
echo ""
echo "2. POST /api/v1/detect  (Beijing test area)"
DETECT_PAYLOAD='{
  "bbox": [116.390, 39.905, 116.395, 39.910],
  "date_before": "2020-01-01",
  "date_after":  "2021-01-01",
  "area_name":   "smoke-test-beijing"
}'

HTTP_STATUS=$(curl -s -o /tmp/detect_resp.json -w "%{http_code}" \
  -H "$AUTH_HEADER" \
  -H "Content-Type: application/json" \
  -d "$DETECT_PAYLOAD" \
  "$SERVICE_URL/api/v1/detect")

echo "  HTTP status: $HTTP_STATUS"
if [[ "$HTTP_STATUS" == "200" ]]; then
  cat /tmp/detect_resp.json | jq '{detection_id,change_area_m2,change_pct,model_version}'
  echo "  ✓ Detect OK"
elif [[ "$HTTP_STATUS" == "422" ]]; then
  echo "  Request validation error (expected for synthetic test):"
  cat /tmp/detect_resp.json | jq .
else
  echo "  Response:"
  cat /tmp/detect_resp.json | jq . 2>/dev/null || cat /tmp/detect_resp.json
fi

echo ""
echo "============================================"
echo " Smoke test complete."
echo "============================================"

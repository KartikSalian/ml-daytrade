#!/bin/bash
# GCP Cloud Run deployment script
# Usage: bash deploy.sh YOUR_GCP_PROJECT_ID

PROJECT_ID=${1:-"your-gcp-project-id"}
SERVICE_NAME="ml-daytrade"
REGION="us-central1"
IMAGE="gcr.io/$PROJECT_ID/$SERVICE_NAME"

echo "Building and pushing image to GCP..."
gcloud builds submit --tag $IMAGE

echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE \
    --platform managed \
    --region $REGION \
    --memory 4Gi \
    --cpu 2 \
    --timeout 300 \
    --set-env-vars FINNHUB_API_KEY=$FINNHUB_API_KEY \
    --set-env-vars ALPACA_API_KEY=$ALPACA_API_KEY \
    --set-env-vars ALPACA_SECRET_KEY=$ALPACA_SECRET_KEY \
    --set-env-vars TRANSFORMERS_OFFLINE=1 \
    --set-env-vars HF_HUB_OFFLINE=1 \
    --no-allow-unauthenticated

echo "Done. Service URL:"
gcloud run services describe $SERVICE_NAME --region $REGION --format "value(status.url)"

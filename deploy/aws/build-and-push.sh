#!/usr/bin/env bash
# Build the Docker image, push to Amazon ECR, and deploy to ECS (Fargate).
set -euo pipefail

REGION="${AWS_DEFAULT_REGION:-us-east-1}"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REPO="$(git rev-parse --show-toplevel 2>/dev/null && cd "$(dirname "$0")/../.." && basename "$PWD" || echo multi-agent-research)"
TAG="${TAG:-$(git rev-parse --short HEAD 2>/dev/null || echo latest)}"
IMAGE="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:${TAG}"

aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

aws ecr create-repository --repository-name "$REPO" --region "$REGION" >/dev/null 2>&1 || true

docker build -f Dockerfile -t "${IMAGE}" ../..
docker push "${IMAGE}"

echo "Deploying ${IMAGE} ..."
aws ecs update-service --cluster multi-agent-cluster --service multi-agent-service \
  --force-new-deployment --region "$REGION" >/dev/null
echo "Done. Service URL: https://$(aws ecs describe-services --cluster multi-agent-cluster --services multi-agent-service --region "$REGION" --query 'services[0].loadBalancers[0].dnsName' --output text)"
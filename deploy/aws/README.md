# AWS Deployment

Two supported paths: **ECS Fargate** (recommended) or **EC2 + systemd**.

## Prerequisites

- AWS CLI configured: `aws configure`
- Docker running
- Secrets stored in AWS Secrets Manager:
  - `multi-agent/OPENAI_API_KEY`
  - `multi-agent/TAVILY_API_KEY`

## Option A — ECS Fargate (recommended)

1. Replace the `ACCOUNT_ID` placeholders in `task-definition.json`.
2. Register the task definition:
   ```bash
   aws ecs register-task-definition --cli-input-json file://deploy/aws/task-definition.json
   ```
3. Create the cluster & service (with an ALB in front of port 8000), then:
   ```bash
   chmod +x deploy/aws/build-and-push.sh
   ./deploy/aws/build-and-push.sh
   ```

The build script builds the image from the repo root, pushes to ECR, and
forces a new ECS deployment.

## Option B — EC2 + systemd

1. Build & push the image (see build script), then install Docker on the EC2 host.
2. `aws ecr get-login-password ... | docker login --username AWS --password-stdin ...`
3. Place secrets in `/etc/multi-agent/.env` (see `.env.example`).
4. Copy the unit file:
   ```bash
   sudo cp deploy/aws/ec2/multi-agent-research.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now multi-agent-research
   ```
5. Health check: `curl http://HOST:8000/api/v1/health`

## Networking

- Expose `/api/v1/webhook/agent` (and `/api/v1/health`) publicly.
- n8n points its HTTP Request node at that webhook URL.
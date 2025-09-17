# Docker Setup Guide

## Overview

This project provides comprehensive Docker support with multi-stage builds, production optimizations, and development convenience features.

## Docker Architecture

### Multi-stage Dockerfile

The Dockerfile uses multi-stage builds for optimized images:

- **base**: Common dependencies and Python setup
- **development**: All dependencies + dev tools + tests
- **production**: Only runtime dependencies, non-root user, health checks

### Image Variants

- `lesson1-mlops:dev` - Development image with all tools
- `lesson1-mlops:0.1` - Production image, optimized for deployment

## Quick Start

### Prerequisites

Install Docker Desktop:
```bash
# macOS
brew install --cask docker

# Ubuntu
sudo apt-get update && sudo apt-get install docker.io docker-compose

# Start Docker service
sudo systemctl start docker
```

### Basic Usage

```bash
# Build production image
make docker-build

# Run training
make docker-run

# Build development image
make docker-build-dev

# Interactive development
make docker-run-dev
```

## Development Workflow

### 1. Development Container

```bash
# Start development container with hot-reload
make docker-run-dev

# Or manually
docker run --rm -it \
  -v "$(PWD)/mlruns:/app/mlruns" \
  -v "$(PWD)/src:/app/src" \
  lesson1-mlops:dev bash
```

### 2. Running Tests in Container

```bash
# Run tests
make docker-test

# Or manually
docker run --rm lesson1-mlops:dev poetry run pytest -v
```

### 3. Interactive Shell

```bash
# Get shell access
make docker-shell

# Execute specific commands
docker exec -it <container_name> python -m src.app.train
```

## Docker Compose

### Services Available

```yaml
# docker-compose.yml provides:
ml-training:    # Production training run
ml-dev:         # Development environment
mlflow-ui:      # MLflow UI on port 5000
jupyter:        # Jupyter Lab on port 8888
```

### Compose Commands

```bash
# Start all services
make compose-up

# Individual services
make compose-up-training    # Just training
make compose-up-mlflow      # MLflow UI
make compose-up-jupyter     # Jupyter Lab
make compose-up-dev         # Development shell

# Management
make compose-down           # Stop all services
make compose-logs           # View logs
make compose-clean          # Complete cleanup
```

### MLflow UI

```bash
# Start MLflow UI in container
make compose-up-mlflow

# Access at http://localhost:5000
```

### Jupyter Lab

```bash
# Start Jupyter Lab
make compose-up-jupyter

# Access at http://localhost:8888
# No token required (development only!)
```

## Production Deployment

### Security Features

- ✅ Non-root user (`appuser`)
- ✅ Minimal dependencies
- ✅ Security updates
- ✅ Health checks
- ✅ No dev tools in production

### Environment Variables

```bash
# Required for production
MLFLOW_TRACKING_URI=file:./mlruns

# Optional
PYTHONUNBUFFERED=1
PYTHONDONTWRITEBYTECODE=1
```

### Volume Mounts

```bash
# MLflow artifacts (persistent storage)
-v /host/mlruns:/app/mlruns

# Configuration override
-v /host/configs:/app/configs
```

## Troubleshooting

### Common Issues

**1. Permission Errors**
```bash
# Fix mlruns permissions
sudo chown -R 1000:1000 mlruns/
```

**2. Docker Build Fails**
```bash
# Clean build without cache
docker build --no-cache -t lesson1-mlops:0.1 .

# Check system resources
docker system df
```

**3. Container Won't Start**
```bash
# Check logs
docker logs <container_id>

# Debug with shell
docker run --rm -it lesson1-mlops:dev bash
```

**4. Poetry Lock Issues**
```bash
# Rebuild poetry.lock in container
docker run --rm -v "$(PWD):/app" lesson1-mlops:dev poetry lock
```

### Performance Optimization

**1. Layer Caching**
```dockerfile
# Dependencies first (changes rarely)
COPY pyproject.toml poetry.lock ./
RUN poetry install

# Code last (changes frequently)
COPY src src
```

**2. Multi-stage Benefits**
- Production: ~200MB smaller
- Development: Full tooling
- Shared base layers

**3. .dockerignore Optimization**
```
# Exclude unnecessary files
mlruns/
data/
.git/
__pycache__/
```

## Advanced Usage

### Custom Networks

```bash
# Create custom network
docker network create ml-network

# Run with custom network
docker run --network ml-network lesson1-mlops:0.1
```

### Resource Limits

```bash
# Limit memory and CPU
docker run --memory=1g --cpus=2 lesson1-mlops:0.1
```

### Health Monitoring

```bash
# Check health status
docker inspect --format='{{.State.Health.Status}}' <container>

# View health logs
docker inspect --format='{{range .State.Health.Log}}{{.Output}}{{end}}' <container>
```

## CI/CD Integration

### GitHub Actions Example

```yaml
# .github/workflows/docker.yml
- name: Build and test
  run: |
    make docker-build-dev
    make docker-test
    make docker-build-prod

- name: Security scan
  run: |
    docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
      aquasec/trivy lesson1-mlops:0.1
```

### Automated Builds

```bash
# Build both variants
make docker-build-dev docker-build-prod

# Tag for registry
docker tag lesson1-mlops:0.1 registry.company.com/ml/lesson1:0.1

# Push to registry
docker push registry.company.com/ml/lesson1:0.1
```

## Best Practices

### Security

- ✅ Use non-root user
- ✅ Pin dependency versions
- ✅ Regular security updates
- ✅ Minimal attack surface
- ❌ Don't store secrets in images

### Performance

- ✅ Multi-stage builds
- ✅ Layer caching optimization
- ✅ Specific base images
- ✅ Remove build dependencies
- ❌ Don't use latest tags

### Maintainability

- ✅ Document all custom configurations
- ✅ Use .dockerignore
- ✅ Health checks
- ✅ Proper logging
- ❌ Don't hardcode paths

## Cleanup Commands

```bash
# Project cleanup
make docker-clean
make compose-clean

# System cleanup
docker system prune -a --volumes
docker builder prune
```

## Monitoring

```bash
# Container stats
docker stats

# System usage
docker system df

# Image sizes
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
```

This Docker setup provides production-ready containerization with development convenience features!

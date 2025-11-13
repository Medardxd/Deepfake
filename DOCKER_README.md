# Docker Deployment Guide

This guide explains how to build and run the Deepfake Detector application using Docker.

## Prerequisites

- Docker installed (version 20.10+)
- Docker Compose installed (version 1.29+)
- At least 8GB of RAM available
- At least 10GB of free disk space (for models and images)

## Quick Start

### Option 1: Using Docker Compose (Recommended)

```bash
# Build and start the application
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the application
docker-compose down
```

Access the application at: http://localhost:8501

### Option 2: Using Docker CLI

```bash
# Build the image
docker build -t deepfake-detector .

# Run the container
docker run -d \
  --name deepfake-detector \
  -p 8501:8501 \
  -v deepfake_models:/root/.cache/huggingface \
  deepfake-detector

# View logs
docker logs -f deepfake-detector

# Stop and remove the container
docker stop deepfake-detector
docker rm deepfake-detector
```

## Configuration

### Environment Variables

You can customize the application by setting environment variables:

```yaml
environment:
  - STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200  # Max upload size in MB
  - STREAMLIT_SERVER_PORT=8501             # Streamlit port
  - STREAMLIT_SERVER_ADDRESS=0.0.0.0       # Listen address
```

### Resource Limits

Adjust CPU and memory limits in `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '4.0'      # Maximum CPUs
      memory: 8G       # Maximum memory
```

### Persistent Model Cache

Models are downloaded on first run and cached in a Docker volume:

```bash
# List volumes
docker volume ls

# Inspect the model cache volume
docker volume inspect deepfake_model_cache

# Remove the volume (models will be re-downloaded)
docker volume rm deepfake_model_cache
```

## First Run

On the first run, the application will download the following models:
- CLIP (openai/clip-vit-base-patch32) - ~600MB
- MTCNN face detector - ~2MB
- Face deepfake detector (prithivMLmods/deepfake-detector-model-v1) - ~400MB
- General AI detector (Ateeqq/ai-vs-human-image-detector) - ~370MB

**Total download size: ~1.4GB**

This may take 5-15 minutes depending on your internet connection.

## Troubleshooting

### Container fails to start

```bash
# Check container logs
docker logs deepfake-detector

# Check if port 8501 is already in use
netstat -an | grep 8501

# Use a different port
docker run -p 8502:8501 deepfake-detector
```

### Out of memory errors

Increase memory limit in `docker-compose.yml` or allocate more RAM to Docker:

```yaml
deploy:
  resources:
    limits:
      memory: 16G  # Increase to 16GB
```

### Models not downloading

Check internet connectivity and firewall settings:

```bash
# Test connection to HuggingFace
docker exec -it deepfake-detector curl -I https://huggingface.co

# Manually trigger model download
docker exec -it deepfake-detector python -c "from transformers import pipeline; pipeline('image-classification', model='Ateeqq/ai-vs-human-image-detector')"
```

### Video processing errors

Ensure FFmpeg is properly installed:

```bash
# Check FFmpeg installation
docker exec -it deepfake-detector ffmpeg -version

# Rebuild the image if needed
docker-compose build --no-cache
```

## Development

To run the application with live code reloading:

```bash
# Mount the current directory as a volume
docker run -d \
  -p 8501:8501 \
  -v $(pwd):/app \
  -v deepfake_models:/root/.cache/huggingface \
  deepfake-detector
```

## Production Deployment

For production deployments, consider:

1. **Reverse Proxy**: Use Nginx or Traefik
2. **HTTPS**: Enable SSL/TLS certificates
3. **Authentication**: Add authentication layer
4. **Resource Monitoring**: Use Docker stats or Prometheus
5. **Logging**: Configure log aggregation
6. **Backups**: Backup the model cache volume

Example with Nginx reverse proxy:

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

## Cleaning Up

```bash
# Stop and remove containers
docker-compose down

# Remove images
docker rmi deepfake-detector

# Remove volumes (including cached models)
docker volume rm deepfake_model_cache

# Remove all unused Docker resources
docker system prune -a
```

## Performance Tips

1. **CPU vs GPU**: This Dockerfile uses CPU-only PyTorch. For GPU support, modify the Dockerfile to use CUDA base images.
2. **RAM**: Allocate at least 8GB RAM for smooth operation
3. **Video size**: Keep videos under 1 minute and 200MB for optimal performance
4. **Concurrent users**: For multiple users, increase resource limits accordingly

## Support

For issues specific to Docker deployment, check:
- Docker logs: `docker logs deepfake-detector`
- Container status: `docker ps -a`
- Resource usage: `docker stats deepfake-detector`

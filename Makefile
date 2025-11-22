# Configuration
MINIO_NAME := minio
MINIO_USER := admin
MINIO_PASS := password
MINIO_PORT := 9000
MINIO_CONSOLE := 9001

.PHONY: minio-up minio-down minio-logs

# Start MinIO (checks if already running to avoid errors)
minio-up:
	@echo "Starting MinIO..."
	@podman rm -f $(MINIO_NAME) 2>/dev/null || true
	@podman run -d --name $(MINIO_NAME) \
		-p $(MINIO_PORT):9000 \
		-p $(MINIO_CONSOLE):9001 \
		-e "MINIO_ROOT_USER=$(MINIO_USER)" \
		-e "MINIO_ROOT_PASSWORD=$(MINIO_PASS)" \
		quay.io/minio/minio server /data --console-address ":9001"

# Teardown
minio-down:
	@podman rm -f $(MINIO_NAME)

# Quick access to logs
minio-logs:
	@podman logs -f $(MINIO_NAME)

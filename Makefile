.PHONY: build test test-postgres test-e2e-docker test-e2e-docker-extended \
       postgres-up postgres-down postgres-restart clean

# Default: build the workspace
build:
	cargo build --workspace

# Tier 1: LiteStore-only tests (no external services required)
test:
	cargo test --workspace

# Tier 2: PostgreSQL tests (starts/stops container automatically)
test-postgres: postgres-up
	@echo "--- Running test suite with PostgreSQL active ---"
	cargo test --workspace --features postgres; status=$$?; $(MAKE) postgres-down; exit $$status

# Tier 4: Docker-based E2E test (real HTTP MCP transport + real ONNX embeddings)
test-e2e-docker:
	@echo "--- Running Docker E2E MCP integration test ---"
	cargo test --test docker_mcp_e2e_test -- --nocapture

test-e2e-docker-extended:
	@echo "--- Running Docker E2E MCP integration test (extended) ---"
	CORVIA_E2E_EXTENDED=1 cargo test --test docker_mcp_e2e_test -- --nocapture

# --- PostgreSQL container management ---

PG_CONTAINER := corvia-postgres-test
PG_IMAGE     := pgvector/pgvector:pg17
PG_PORT      := 5432

postgres-up:
	@if docker ps --format '{{.Names}}' | grep -q '^$(PG_CONTAINER)$$'; then \
		echo "PostgreSQL already running ($(PG_CONTAINER))"; \
	else \
		echo "Starting PostgreSQL on port $(PG_PORT)..."; \
		docker run -d --rm \
			--name $(PG_CONTAINER) \
			-p $(PG_PORT):5432 \
			-e POSTGRES_USER=corvia \
			-e POSTGRES_PASSWORD=corvia \
			-e POSTGRES_DB=corvia \
			$(PG_IMAGE); \
		sleep 3; \
		echo "PostgreSQL ready at 127.0.0.1:$(PG_PORT)"; \
	fi

postgres-down:
	@if docker ps --format '{{.Names}}' | grep -q '^$(PG_CONTAINER)$$'; then \
		echo "Stopping PostgreSQL..."; \
		docker stop $(PG_CONTAINER) > /dev/null 2>&1 || true; \
		echo "PostgreSQL stopped"; \
	else \
		echo "PostgreSQL not running"; \
	fi

postgres-restart: postgres-down
	@sleep 1
	@$(MAKE) postgres-up

clean:
	cargo clean

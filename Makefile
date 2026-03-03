.PHONY: build test test-full test-postgres test-all \
       surrealdb-up surrealdb-down surrealdb-restart \
       postgres-up postgres-down postgres-restart clean

# Default: build the workspace
build:
	cargo build --workspace

# Tier 1: LiteStore-only tests (no external services required)
test:
	cargo test --workspace

# Tier 2: Full test suite with SurrealDB (starts/stops container automatically)
test-full: surrealdb-up
	@echo "--- Running full test suite (SurrealDB active) ---"
	cargo test --workspace; status=$$?; $(MAKE) surrealdb-down; exit $$status

# Tier 2.5: PostgreSQL tests (starts/stops container automatically)
test-postgres: postgres-up
	@echo "--- Running test suite with PostgreSQL active ---"
	cargo test --workspace --features postgres; status=$$?; $(MAKE) postgres-down; exit $$status

# Full: SurrealDB + PostgreSQL tests together
test-all: surrealdb-up postgres-up
	@echo "--- Running all tests (SurrealDB + PostgreSQL active) ---"
	cargo test --workspace --features postgres; status=$$?; \
		$(MAKE) surrealdb-down; $(MAKE) postgres-down; exit $$status

# --- SurrealDB container management ---

SURREAL_CONTAINER := corvia-surrealdb-test
SURREAL_IMAGE     := surrealdb/surrealdb:v3
SURREAL_PORT      := 8000

surrealdb-up:
	@if docker ps --format '{{.Names}}' | grep -q '^$(SURREAL_CONTAINER)$$'; then \
		echo "SurrealDB already running ($(SURREAL_CONTAINER))"; \
	else \
		echo "Starting SurrealDB on port $(SURREAL_PORT)..."; \
		docker run -d --rm \
			--name $(SURREAL_CONTAINER) \
			-p $(SURREAL_PORT):8000 \
			$(SURREAL_IMAGE) \
			start --log=warn --user root --pass root; \
		sleep 2; \
		echo "SurrealDB ready at 127.0.0.1:$(SURREAL_PORT)"; \
	fi

surrealdb-down:
	@if docker ps --format '{{.Names}}' | grep -q '^$(SURREAL_CONTAINER)$$'; then \
		echo "Stopping SurrealDB..."; \
		docker stop $(SURREAL_CONTAINER) > /dev/null 2>&1 || true; \
		echo "SurrealDB stopped"; \
	else \
		echo "SurrealDB not running"; \
	fi

surrealdb-restart: surrealdb-down
	@sleep 1
	@$(MAKE) surrealdb-up

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

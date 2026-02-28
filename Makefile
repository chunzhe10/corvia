.PHONY: build test test-full surrealdb-up surrealdb-down surrealdb-restart clean

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

clean:
	cargo clean

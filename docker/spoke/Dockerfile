# corvia spoke container
# A lightweight container for running Claude Code against a shared corvia hub.
#
# Build:
#   docker build -t corvia-spoke:latest --build-arg CLAUDE_CODE_VERSION=1.0.18 docker/spoke/
#
# The spoke connects to a corvia hub server via MCP for shared organizational memory.
# Auth credentials and MCP token are injected at runtime, not baked into the image.

FROM node:22-bookworm

# System dependencies (no build-essential: Claude Code is pure JS, no native addons)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        jq \
    && rm -rf /var/lib/apt/lists/*

# Install GitHub CLI
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        -o /usr/share/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends gh && \
    rm -rf /var/lib/apt/lists/*

# Pin Claude Code version -- never use @latest at runtime
ARG CLAUDE_CODE_VERSION=1.0.18
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}
LABEL corvia.spoke.claude_code_version=${CLAUDE_CODE_VERSION}

# Create non-root user for security
RUN useradd -m -s /bin/bash spoke

# Copy entrypoint (before USER switch so it lands in root-owned /usr/local/bin)
COPY --chown=spoke:spoke spoke-entrypoint.sh /usr/local/bin/spoke-entrypoint.sh

USER spoke
WORKDIR /home/spoke

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD pgrep -x "claude" || exit 1

ENTRYPOINT ["/usr/local/bin/spoke-entrypoint.sh"]

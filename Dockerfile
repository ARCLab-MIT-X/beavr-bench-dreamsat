# Optimized Dockerfile for beavr-bench
# Inspired by: https://github.com/mujocolab/mjlab/blob/main/Dockerfile

FROM nvidia/cuda:12.8.0-devel-ubuntu24.04
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git \
    curl \
    libegl-dev \
    libgl1 \
    libosmesa6 \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_PREFERENCE=only-managed

# Use Python 3.12 as per pyproject.toml requirement
RUN uv python install 3.12

WORKDIR /app

# Install dependencies first for better caching
COPY uv.lock pyproject.toml /app/
RUN uv sync --locked --no-install-project --no-editable --no-dev

# Add the rest of the application
COPY . /app

# Sync the project
RUN uv sync --locked --no-editable --no-dev

# Ensure scripts are available in the PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV MUJOCO_GL=egl

# Default command: run tests
CMD ["uv", "run", "pytest"]

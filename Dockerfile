# CPU-first development image for local work and future Colab GPU training.
FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies needed for package installation and basic tooling.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip before installing project dependencies.
RUN pip install --upgrade pip

# Copy dependency manifest first for better layer caching.
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project into the container.
COPY . ./

# Default shell entrypoint for manual script execution.
CMD ["bash"]

FROM python:3.11-slim

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     PORT=5050

# Install essential system runtime libraries for video decoding and processing
RUN apt-get update && apt-get install -y --no-install-recommends     ffmpeg     libglib2.0-0     libsm6     libxext6     libxrender-dev     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install CPU-optimized PyTorch and dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Expose dashboard port
EXPOSE 5050

# Launch the unified Flask dashboard
CMD ["python", "app.py"]

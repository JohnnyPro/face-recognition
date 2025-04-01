# Use official Python image as base
FROM python:3.11-slim


# Set the working directory
WORKDIR /app

# Copy the required files
COPY main-requirements.txt main-requirements.txt

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    build-essential \
    libglib2.0-0 \
    g++ \
    && rm -rf /var/lib/apt/lists/*
# Install dependencies
RUN pip install --no-cache-dir -r main-requirements.txt

# Copy the application code
COPY . .

# Expose the FastAPI default port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


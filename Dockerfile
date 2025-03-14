# Use official Python image as base
FROM python:3.11

# Set the working directory
WORKDIR /app

# Copy the required files
COPY app/requirements.txt /app/requirements.txt

RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the FastAPI default port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


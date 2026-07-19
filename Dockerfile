# This Dockerfile is used to create a Docker image for a Python application.
# Base image is Python 3.10 slim version to keep the image size small.
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies specified in requirements.txt
RUN pip install --no-cache-dir \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    -r requirements.txt

# Copy the rest of the application code into the container
COPY api/ ./api/
COPY models/ ./models/

# Expose the port that the application will run on
EXPOSE 8000

# Run the application using uvicorn when the container starts
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]


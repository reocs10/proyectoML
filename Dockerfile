# Use an official Python runtime as the base image
FROM python:3.10.12

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model and scaler files into the container
COPY optimal_rf_model.pkl .
COPY scaler.pkl .

# Copy the Python script into the container
COPY main.py .

# Set the default command to execute when the container starts

ENTRYPOINT ["python", "main.py"]

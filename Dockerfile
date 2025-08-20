# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies first
# This is a Docker best practice for caching layers
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
# This includes app.py, templates/, Run_Predictions/, XML_to_dat/, etc.
COPY . .

# Expose the port the app will run on
EXPOSE 5000

# Define the command to run your app using Gunicorn
CMD ["python", "app.py"]

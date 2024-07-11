# Use the official Python image from the Docker Hub
FROM python:3.9

# Set the working directory in the container
WORKDIR /

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the dependencies specified in requirements.txt
RUN pip install -r requirements.txt

# Copy the rest of the application code to the working directory
COPY . .

# Expose the port that Streamlit runs on
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "./app/app.py"]

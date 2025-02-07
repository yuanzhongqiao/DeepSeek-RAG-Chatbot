# Use an official Python runtime as a parent image (change 'buster' to 'slim', if needed)
FROM python:3.9-buster  # You can adjust the version according to your requirements or base on what is available in Docker Hub for Python images.
LABEL maintainer="your_name@example.com"  # Replace with actual contact information, GitHub profile URL, etc., using LABEL instruction (optional).

WORKDIR /usr/src/app  # Set the working directory inside the container to where your app will be installed and run from.

COPY requirements.txt ./  # Copy over any dependencies file needed for installation of Python packages within this folder in Docker image.
RUN pip install --no-cache-dir -r requirements.txt  # Install any necessary Python package(s) listed as dependencies by running a single command on the list inside 'requirements.txt'.

COPY . .  # Copy everything else from your current directory into the container's working directory (e.g., app files, etc.).

EXPOSE 8501  # Expose port for Streamlit app, which is defaulted here but can be changed depending upon your application setup.

CMD ["streamlit", "run", "app.py"]  # This tells Docker what to run when starting up our image using docker run command later on after building it with 'docker build -t chatbot .'.


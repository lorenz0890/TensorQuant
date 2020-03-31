FROM nvcr.io/nvidia/tensorflow:20.01-tf2-py3
# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

RUN apt-get update
RUN apt-get install -y ca-certificates

# Add the dependencies to the container and install the python dependencies
RUN pip3 install Pillow

WORKDIR ./TensorQuant/Kernels
RUN make all
WORKDIR /app

ENV PYTHONPATH=${PYTHONPATH}:./
CMD ["python", "./TensorQuant/Experiments/AlexNet/main.py"]

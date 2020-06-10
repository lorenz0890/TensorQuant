FROM nvcr.io/nvidia/tensorflow:20.03-tf2-py3
# Set the working directory to /app


#RUN apt-get update && \
#       apt-get -y install sudo
#RUN useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo
#USER docker


WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

RUN apt-get update
RUN apt-get install -y ca-certificates

#RUN apt-get install graphviz -y

# Add the dependencies to the container and install the python dependencies
#RUN pip3 install Pillow
#RUN pip3 install pydotplus
#RUN pip3 install pydot
#RUN pip3 install graphviz

#RUN pip3 install virtualenv
RUN pip3 install tensorflow-gpu==2.2.0
#RUN export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

WORKDIR ./TensorQuant/Kernels
RUN make all
WORKDIR /app

ENV LD_LIBRARY_PATH=:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
ENV LD_INCLUDE_PATH=:/usr/local/cuda/include:/usr/local/cuda/extras/CUPTI/include
ENV PYTHONPATH=${PYTHONPATH}:./
#ADD start.sh /
#RUN chmod +x /start.sh
#RUN mkdir profile_env
#CMD ["/start.sh"]

RUN mkdir /usr/local/cuda/extras/CUPTI/lib64
RUN ln -s /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcupti.so.10.2 /usr/local/cuda/extras/CUPTI/lib64/libcupti.so.10.1

#RUN rm /usr/local/cuda/lib64/libcudart.so.10.2
RUN ln -s /usr/local/cuda/lib64/libcudart.so.10.2 /usr/local/cuda/lib64/libcudart.so.10.1
#RUN ln -s /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcudart.so.10.2 /usr/local/cuda/lib64/libcudart.so.10.1
#RUN ln -s /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcudart.so.10.2 /usr/local/cuda/lib64/libcudart.so.10.2

#CMD /bin/bash
CMD ["python", "./TensorQuant/Examples/LeNet/main.py"]

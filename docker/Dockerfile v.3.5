# Use an official Python runtime as a parent image
FROM python:3.6.6

# Set the working directory to /qlik-py-tools
WORKDIR /qlik-py-tools

# Copy the current directory contents into the container at /qlik-py-tools
COPY . /qlik-py-tools

# Install dependencies
RUN apt-get update
RUN apt-get install build-essential
	
# Upgrade pip and setuptools
RUN python -m pip install --upgrade setuptools pip

# Install required packages
RUN pip install grpcio
RUN pip install grpcio-tools
RUN pip install numpy
RUN pip install scipy
RUN pip install pandas
RUN pip install cython
RUN pip install pystan==2.17
RUN pip install fbprophet
RUN pip install -U scikit-learn
RUN pip install hdbscan
RUN pip install -U skater

# Copy modified file for skater
COPY ./feature_importance.py /usr/local/lib/python3.6/site-packages/skater-1.1.2-py3.6.egg/skater/core/global_interpretation/

# Make port 80 available to the world outside this container
EXPOSE 80

# Set the working directory to /qlik-py-tools/core
WORKDIR /qlik-py-tools/core

# Run __main__.py when the container launches
CMD ["python", "__main__.py"]
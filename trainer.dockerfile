# Base image
FROM python:3.9-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# copy relevant directories and files
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/

# set working directory
WORKDIR /

# install python packages
RUN pip install -r requirements.txt --no-cache-dir

# ETRISUB Project for Human and Vehicle Attributes Recognition, Re-ID

## Introduction

The ETRISUB project is a multi-camera control system designed for real-time human and vehicle attribute recognition and re-identification (Re-ID). Cameras capture images and metadata which are sent through
an Apache Kafka pipeline. A **producer** component ingests camera data and publishes messages to Kafka topics. A **consumer** component subscribes to those topics,
processes the incoming data (e.g. human and vehicle attributes extraction, Re-ID), and indexes the results into Elasticsearch for search and analytics.

This architecture enables scalable, distributed processing of surveillance data, with Kafka providing durable messaging and Elasticsearch offering powerful
search capabilities.

## Prerequisites

- Python 3.10 with Conda
- Apache Kafka
- Elasticsearch
- GStreamer (for RTSP camera ingestion)

## Installation

### 1. Clone the repository

Download the source code from GitHub and switch to the `vehicle-api-integrated` branch, which includes vehicle counting integration (the `main` branch contains human analysis only):

```bash
git clone https://github.com/itvkist/etrisub-vision-system.git
cd etrisub-vision-system
git checkout vehicle-api-integrated
```

### 2.1. Install Apache Kafka

Download Kafka from the official website: https://kafka.apache.org/downloads

Extract to the `kafka/` directory in the project:
```bash
tar -xzf kafka_2.13-3.x.x.tgz
mv kafka_2.13-3.x.x kafka
```

Grant execute permissions to Kafka scripts:
```bash
chmod +x kafka/bin/*
```

### 2.2. Install Elasticsearch

Download Elasticsearch from the official website: https://www.elastic.co/downloads/elasticsearch

Extract to the `elasticsearch/` directory in the project:
```bash
tar -xzf elasticsearch-8.x.x-linux-x86_64.tar.gz
mv elasticsearch-8.x.x elasticsearch
```

Grant execute permissions:
```bash
chmod +x elasticsearch/bin/elasticsearch
```

### 2.3. Install the Python environment (etri_gst)

The main Python environment is used for the Consumer (AI processing) and all system services:

```bash
conda create --name etri_gst python=3.10
conda activate etri_gst
conda install -c conda-forge gstreamer pygobject gst-plugins-base gst-plugins-good gst-plugins-bad gst-plugins-ugly gst-libav
pip install -r requirements.txt
```

## Configuration

### Kafka Configuration

- Configuration files are located in `kafka/config/`.
- Default settings work for basic setup.
- Modify `server.properties` and `zookeeper.properties` if needed.

### Elasticsearch Configuration

- Configuration files are located in `elasticsearch/config/`.
- Default settings should be sufficient.

## Usage

### Starting all services

```bash
chmod +x start_services.sh
./start_services.sh
```

This script will:

- Start Zookeeper
- Start Kafka Server
- Start Elasticsearch
- Start the Python Producer
- Start the Python Consumer

### Manual startup

Open separate terminals for each service:

1. **Start Zookeeper**
   ```bash
   cd kafka
   bin/zookeeper-server-start.sh config/zookeeper.properties
   ```
2. **Start Kafka**
   ```bash
   cd kafka
   bin/kafka-server-start.sh config/server.properties
   ```
3. **Start Elasticsearch**
   ```bash
   cd elasticsearch
   ./bin/elasticsearch
   ```
4. **Start Producer**
   ```bash
   conda activate etri_gst
   python producer_gst.py
   ```
5. **Start Consumer**
   ```bash
   conda activate etri_gst
   python consumer.py
   ```

# ETRISUB Project for Human and Vehicle Attributes Recognition, Re-ID

## Introduction

The ETRISUB project is a multi-camera control system designed for real-time human and vehicle attribute recognition and re-identification (Re-ID). Cameras capture images and metadata which are sent through
an Apache Kafka pipeline. A **producer** component ingests camera data and publishes messages to Kafka topics. A **consumer** component subscribes to those topics,
processes the incoming data (e.g. human and vehicle attributes extraction, Re-ID), and indexes the results into Elasticsearch for search and analytics.

This architecture enables scalable, distributed processing of surveillance data, with Kafka providing durable messaging and Elasticsearch offering powerful
search capabilities.

## Prerequisites

- Python 3.10 with pip
- Apache Kafka
- Elasticsearch

## Installation

1. **Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Apache Kafka**
   - Download from the official website.
   - Extract to the `kafka` directory in your project.
   - Ensure Kafka scripts are executable:
     ```bash
     chmod +x kafka/bin/*
     ```

3. **Elasticsearch**
   - Download from the official website.
   - Extract to the `elasticsearch` directory in your project.
   - Ensure the Elasticsearch binary is executable:
     ```bash
     chmod +x elasticsearch/bin/elasticsearch
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
   python producer.py
   ```
5. **Start Consumer**
   ```bash
   python consumer.py
   ```

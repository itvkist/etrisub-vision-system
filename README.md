# Multi camera control system
This project implements a real-time data processing pipeline using Apache Kafka, Elasticsearch, and Python. The system consists of producers that sends data to Kafka and a consumer that processes the data and stores it in Elasticsearch.
# Prerequisites

Python 3.10 with pip
Apache Kafka
Elasticsearch


# Installation

Install Python dependencies
bashCopypip install -r requirements.txt

Install Apache Kafka

Download Apache Kafka from the official website
Extract it to the kafka directory in your project
Ensure the Kafka scripts have execute permissions:
bashCopychmod +x kafka/bin/*



Install Elasticsearch

Download Elasticsearch from the official website
Extract it to the elasticsearch directory in your project
Ensure the Elasticsearch binary has execute permissions:
bashCopychmod +x elasticsearch/bin/elasticsearch




# Configuration

Kafka Configuration

Configuration files are located in kafka/config/
Default configurations should work for basic setup
Modify server.properties and zookeeper.properties if needed


Elasticsearch Configuration

Configuration files are located in elasticsearch/config/
Default configurations should work for basic setup



# Usage

Start all services
bashCopychmod +x start_services.sh
./start_services.sh
This script will:

Start Zookeeper
Start Kafka Server
Start Elasticsearch
Start the Python Producer
Start the Python Consumer


Manual startup (if needed)
You can also start each service manually in separate terminals:
## Terminal 1 - Start Zookeeper
cd kafka
bin/zookeeper-server-start.sh config/zookeeper.properties

## Terminal 2 - Start Kafka
cd kafka
bin/kafka-server-start.sh config/server.properties

## Terminal 3 - Start Elasticsearch
cd elasticsearch
./bin/elasticsearch

## Terminal 4 - Start Producer
python producer.py

## Terminal 5 - Start Consumer
python consumer.py

#!/bin/bash



# Start Zookeeper in a new terminal
cd kafka
echo "Starting Zookeeper..."
gnome-terminal -- bash -c "cd $(pwd); bin/zookeeper-server-start.sh config/zookeeper.properties; exec bash"
sleep 5  # Wait for Zookeeper to start

# Start Kafka Server in a new terminal 
echo "Starting Kafka server..."
gnome-terminal -- bash -c "cd $(pwd); bin/kafka-server-start.sh config/server.properties; exec bash"
sleep 5  # Wait for Kafka server to start

# Start Elasticsearch in a new terminal
cd ../elasticsearch
echo "Starting Elasticsearch..."
gnome-terminal -- bash -c "cd $(pwd); ./bin/elasticsearch; exec bash"
sleep 10  # Wait for Elasticsearch to start

# Start Python Producer in a new terminal
cd ..
echo "Starting Python Producer..."
gnome-terminal -- bash -c "cd $(pwd); python producer.py; exec bash"
sleep 2

# Start Python Consumer in a new terminal
echo "Starting Python Consumer..."
gnome-terminal -- bash -c "cd $(pwd); python app.py; exec bash"
sleep 2

echo "All services started successfully!"
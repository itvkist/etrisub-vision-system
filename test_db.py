#!/usr/bin/env python3

from elasticsearch import Elasticsearch
import json

def test_elasticsearch():
    # Connect to Elasticsearch
    es = Elasticsearch(
            hosts=[{'host': 'localhost', 'port': 9200, 'scheme': 'http'}],
            # Alternative format:
            # hosts=['http://localhost:9200'],
        )
    
    try:
        # Test connection
        if es.ping():
            print("✅ Successfully connected to Elasticsearch!")
            
            # Get cluster info
            info = es.info()
            print(f"Cluster name: {info['cluster_name']}")
            print(f"Elasticsearch version: {info['version']['number']}")
            
            # Create a test index and document
            doc = {
                'title': 'Test Document',
                'content': 'This is a test document for Elasticsearch',
                'timestamp': '2024-01-01T00:00:00'
            }
            
            # Index the document
            result = es.index(index='test-index', id=1, body=doc)
            print(f"✅ Document indexed: {result['result']}")
            
            # Search for the document
            search_result = es.search(
                index='test-index',
                body={
                    'query': {
                        'match': {
                            'content': 'test'
                        }
                    }
                }
            )
            
            if search_result['hits']['total']['value'] > 0:
                print("✅ Search successful!")
                print(f"Found {search_result['hits']['total']['value']} document(s)")
                
                for hit in search_result['hits']['hits']:
                    print(f"Document ID: {hit['_id']}")
                    print(f"Source: {hit['_source']}")
            
            # Clean up - delete the test index
            es.indices.delete(index='test-index')
            print("✅ Test index cleaned up")
            
        else:
            print("❌ Failed to connect to Elasticsearch")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure Elasticsearch is running on localhost:9200")

if __name__ == "__main__":
    test_elasticsearch()
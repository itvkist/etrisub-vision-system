import urllib3
from elasticsearch import Elasticsearch

class ElasticsearchClient:
    def __init__(self):
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        self.es = Elasticsearch(
            hosts=[{'host': 'localhost', 'port': 9200, 'scheme': 'http'}],
            # Alternative format:
            # hosts=['http://localhost:9200'],
        )
    
    def index_analysis(self, doc):
        return self.es.index(index='surveillance_analysis', body=doc)
    
    def search(self, search_term):
        try:
            query = {
            "query": {
                "multi_match": {
                    "query": search_term,
                    "fields": ["filtered_content", "attributes.*.*.keyword"]
                }
            }
        }
            
            response = self.es.search(index="surveillance_analysis", body=query)
            hits = response.get('hits', {}).get('hits', [])
            return [hit['_source'] for hit in hits]
            
        except Exception as e:
            print(f"Search error details: {str(e)}")
            return []
    
    def delete_index(self,doc):
        self.es.indices.delete(index=doc,ignore=[400,404])
        
    def create_index(self,doc):
        self.es.indices.create(
            index='surveillance_analysis',
            body={
                'mappings': {
                    'properties': {
                        'timestamp': {'type': 'date'},
                        'image': {'type': 'text'},
                        'filtered_content': {'type': 'text', 'analyzer': 'standard'},
                        'filtered_content_vi': {'type': 'text', 'analyzer': 'standard'},
                        'attributes': {
                            'type': 'nested',
                            'properties': {
                                'type': {'type': 'keyword'},
                                'color': {'type': 'keyword'}
                            }
                        },
                        'camera_id': {'type': 'integer'}
                    }
                }
            }
        )
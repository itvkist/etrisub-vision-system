from elastic_client import ElasticsearchClient

def test_search_cli(query):
    client = ElasticsearchClient()
    results = client.search(query)
    for result in results:
        print(result)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python test_search_cli.py <search_term>")
    else:
        search_term = sys.argv[1]
        test_search_cli(search_term)
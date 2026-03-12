import requests
from urllib.parse import urlparse

def make_request(url, method='GET', **kwargs):
    """
    Make HTTP/HTTPS request with automatic protocol fallback
    """
    parsed_url = urlparse(url)
    
    # Try HTTPS first
    try:
        if not parsed_url.scheme:
            url = f'https://{url}'
        response = requests.request(method, url, **kwargs)
        return response
    except requests.exceptions.SSLError:
        # If HTTPS fails, try HTTP
        http_url = url.replace('https://', 'http://', 1)
        if http_url == url:
            http_url = f'http://{url}'
        return requests.request(method, http_url, **kwargs)

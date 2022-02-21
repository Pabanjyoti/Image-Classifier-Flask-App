import requests

def valid_url(url):
    
    response = requests.get(url)
    mimetype = response.headers.get("Content-Type", default = None)

    if mimetype:
        return any([mimetype.startswith("image")])
    else:
        return False
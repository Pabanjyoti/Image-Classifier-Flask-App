import requests

def valid_url(url):
    
    response = requests.get(url)
    mimetype = response.headers.get("Content-Type", default = None)

    if any([mimetype.startswith("image")]) == True:
        return mimetype
    else:
        return 'invalid_image_url'
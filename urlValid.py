import mimetypes

def valid_url(url):

    mimetype, encoding = mimetypes.guess_type(url)
    
    if mimetype:
        return any([mimetype.startswith("image")])
    else:
        return False
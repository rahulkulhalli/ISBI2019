import os
import requests
import json
from tqdm import tqdm


def download_model():
    if not os.path.exists(os.path.join(os.getcwd(), "assets", "model_url.json")):
        raise IOError("Cannot download model. model_url file is missing from assets")

    json_file = None
    try:
        with open(os.path.join(os.getcwd(), "assets", "model_url.json"), 'r') as readfile:
            json_file = json.load(readfile)
    except IOError:
        print("Error reading JSON.")

    if json_file is None or "url" not in json_file.keys():
        raise AttributeError("Something went wrong while parsing the JSON.")

    url = None
    name = None
    try:
        url = json_file["url"]
        name = json_file["model_name"]
    except IOError:
        print("Error while accessing key(s) in JSON.")

    dst = os.path.join(os.getcwd(), "assets", name)

    if os.path.exists(dst):
        print("File already exists. Not downloading again.")
        print(50*'-')
        return dst

    try:

        print("Downloading model...")

        file_size = int(requests.head(url).headers["Content-Length"])
        if os.path.exists(dst):
            first_byte = os.path.getsize(dst)
        else:
            first_byte = 0
        if first_byte >= file_size:
            return file_size
        header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}
        pbar = tqdm(
            total=file_size, initial=first_byte,
            unit='B', unit_scale=True, desc="ResNeXt50.h5")
        req = requests.get(url, headers=header, stream=True)
        with(open(dst, 'ab')) as f:
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(1024)

        if f is not None:
            f.close()

        pbar.close()

        print("Downloaded model at {}.".format(dst))
        print(50*'-')

    except Exception as e:
        print("Something went wrong while downloading the model.")
        print(e)

    return dst

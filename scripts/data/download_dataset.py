import requests

# URL for dataset (zip file)
url = 'https://www.dropbox.com/scl/fi/n02gzmthp2f9tntgpwsdh/species_data_original.zip?rlkey=sf1hj0o8fqd7fer32eueiicg1&st=z2l75xf8&dl=1'

# Download zip file
r = requests.get(url, stream=True)
with open('species_data_original.zip', 'wb') as fd:
    for chunk in r.iter_content(chunk_size=128):
        fd.write(chunk)
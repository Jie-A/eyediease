import requests
import urllib3
from pathlib import Path
urllib3.disable_warnings()


path = Path('/home/.jarviscloud/jarvisconfig')


def pause():
    token, id = path.open().read().split()
    requests.post('https://backend.jarvislabs.ai:8000/pause',
                  json={'jwt': token, 'id': id}, verify=False)

def test():
    token, id = path.open().read().split()
    print(token)
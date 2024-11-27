import requests
import time

def main():
    url = 'http://localhost:5000/simplify'
    path = '../documentos/acordaos/0600012-49_REl_28052024_1.pdf'
    with open(path, 'rb') as f:
        pdf = f.read()
    sections = [0, 2, 5, 10, 12]

    a = time.monotonic()
    res = requests.post(url, files={'doc': pdf}, data={'sections': sections})
    b = time.monotonic()
    duration_s = b - a
    print(f"Time: {b - a} seconds")
    print(res.json())

if __name__ == '__main__':
    main()

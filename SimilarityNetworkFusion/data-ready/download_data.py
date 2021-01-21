import os
import requests
import zipfile
import shutil

"""
  'https://doc-0s-bs-docs.googleusercontent.com/docs/securesc/4epkg8quepvsrhdum7mask9mon729f35/oppbq4ssbt8gk9qqvci112s1q597vi4n/1611216375000/00144453899101048022/00144453899101048022/1KnPk5VQgfRMrQ9B1BH-uKN_mruCU4frC?e=download&authuser=0&nonce=kk6cbqqbnfdak&user=00144453899101048022&hash=lvsddiehuiiabqmbs70qu5b5sj60lv74' 
"""

if __name__ == '__main__':
    url = 'https://doc-0s-bs-docs.googleusercontent.com/docs/securesc/4epkg8quepvsrhdum7mask9mon729f35/oppbq4ssbt8gk9qqvci112s1q597vi4n/1611216375000/00144453899101048022/00144453899101048022/1KnPk5VQgfRMrQ9B1BH-uKN_mruCU4frC?e=download&authuser=0&nonce=kk6cbqqbnfdak&user=00144453899101048022&hash=lvsddiehuiiabqmbs70qu5b5sj60lv74'
    name_file = './data-ready.zip'

    print('Downloading the file...', end='')

    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'it-IT,it;q=0.8,en-US;q=0.5,en;q=0.3',
        'Cookie': 'AUTH_7ofph1k66a5q8crlckh23na95mftot9p_nonce=kk6cbqqbnfdak',
        'DNT': '1',
        'Upgrade-Insecure-Requests': '1',
    }

    with requests.get(url, stream=True, headers=headers) as r:
        with open(name_file, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    print('DONE')
    print('Extracting the file...', end='')
    with zipfile.ZipFile(name_file, 'r') as zip_ref:
        zip_ref.extractall('./')
    os.remove(name_file)
    print('END')
    print('Bye!')
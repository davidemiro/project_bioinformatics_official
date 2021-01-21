import os
import requests
import zipfile
import shutil

if __name__ == '__main__':
    url = "https://uc14d3a001d2691e94f9f81c51c2.dl.dropboxusercontent.com/cd/0/get/BHV6MkmGpC4HA0oOUD_uDpBmQ4lEn5NXn-H8Gv8IZw_sGWNlm6DGIBkxzPUVVe5Vc6xfaQNW5pp7q2Zys-HvZ5aVd0sfBFYiqedSZ-wAAOetdJwPtPrIsNjRM75H1KBkEPE/file?dl=1#"
    name_file = './data-ready.zip'

    print('Downloading the file...', end='')

    with requests.get(url, stream=True) as r:
        with open(name_file, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    print('DONE')
    print('Extracting the file...', end='')
    with zipfile.ZipFile(name_file, 'r') as zip_ref:
        zip_ref.extractall('./')
    os.remove(name_file)
    print('END')
    print('Bye!')

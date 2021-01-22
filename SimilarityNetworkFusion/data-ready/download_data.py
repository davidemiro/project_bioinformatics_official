import os
import requests
import zipfile
import shutil

def download_file_from_google_drive(id, destination):
    URL = "https://drive.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


if __name__ == '__main__':
    url ='https://drive.google.com/file/d/1e5jP45khl0NxX-vrXlqrJHekbCuSJYYu/view?usp=sharing'
    name_file = './data-ready.zip'

    print('Downloading the file...', end='')

    file_id = '1e5jP45khl0NxX-vrXlqrJHekbCuSJYYu'
    destination = './data-ready.zip'
    download_file_from_google_drive(file_id, destination)


    # with requests.get(url, stream=True, headers=headers) as r:
    #     with open(name_file, 'wb') as f:
    #         shutil.copyfileobj(r.raw, f)
    print('DONE')
    print('Extracting the file...', end='')
    with zipfile.ZipFile(name_file, 'r') as zip_ref:
        zip_ref.extractall('./')
    os.remove(name_file)
    print('END')
    print('Bye!')

import os
import zipfile


def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file),
                       os.path.relpath(os.path.join(root, file),
                                       os.path.join(path, '..')))

def zipfolder(path:str,name:str):
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as zip_file:
        zipdir(path,zip_file)
        zip.close()
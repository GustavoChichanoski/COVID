import os
import zipfile


def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file),
                       os.path.relpath(os.path.join(root, file),
                                       os.path.join(path, '..')))

def zipfolder(path:str,name:str):
    zipf = zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED)
    zipdir(path,zipf)
    zip.close()
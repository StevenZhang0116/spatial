import shutil

def delete_pycache(directory):
    for root, dirs, files in os.walk(directory):
        if '__pycache__' in dirs:
            print(f"Deleting: {os.path.join(root, '__pycache__')}")
            shutil.rmtree(os.path.join(root, '__pycache__'))

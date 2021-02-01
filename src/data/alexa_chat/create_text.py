import json
import os
from pathlib import Path

def extract_message(file):
    with open(os.path.join(Path(__file__).parent,file),'r') as f:
        data = f.read()
    json_data = json.loads(data)
    print(type(json_data))
    hash_keys = json_data.keys()
    text = ''.join([f"{content['message']}\n" for key in hash_keys for content in json_data[key]['content']])
    f_name  = file.split('.')[0]
    with open(os.path.join(Path(__file__).parent,'..',f_name),'w') as f:
        f.write(text)

if __name__=="__main__":
    for file in os.listdir(Path(__file__).parent):
        if file.endswith('.json'):
            extract_message(file)
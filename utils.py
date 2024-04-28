import argparse, os, yaml
from huggingface_hub import hf_hub_download

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def namespace2dict(config):
    return vars(config)

def read_yaml(pth):
    with open(pth, 'r') as f:
        config = yaml.safe_load(f)
    return config

def download_data_hf(repo_id="Tracygu/dnscustom", 
                     filename="testset.tar.gz",
                     local_dir="./data"):

    file_path = hf_hub_download(repo_id=repo_id, 
                                filename=filename, 
                                repo_type="dataset",
                                local_dir=local_dir)
    print(f"File has been downloaded and is located at {file_path}")
    return file_path

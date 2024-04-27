import argparse, os, yaml



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
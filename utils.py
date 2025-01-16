import yaml

def config():
    with open('config.yaml', 'r', encoding='utf-8') as file:
        cfg = yaml.safe_load(file)
    return cfg
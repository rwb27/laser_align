import yaml
from pprint import pprint
with open('config.yaml', 'r') as f:
    doc = yaml.load(f)
    pprint(doc)
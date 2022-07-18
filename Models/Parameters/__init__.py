from .hyper import *
import yaml

def hyper_loader(hyper_name = 'gnn_hyper', hyper_path = './Models/Parameters/hyper'):

    file = open(f'{hyper_path}/{hyper_name}.yaml')
    return yaml.load(file.read())


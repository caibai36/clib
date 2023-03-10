from typing import List, Dict, Tuple, Union, Any

import os
import re # parser class name
import json # for data files
import yaml # for config files

import torch
from torch import nn

def load_model_config(model_config: Dict) -> object:
     """ Get a model object from model configuration file.

     The configuration contains model class name and object parameters,
     e.g., {'class': "<class 'seq2seq.asr.EncRNNDecRNNAtt'>", 'dec_embedding_size: 6}
     """
     import importlib
     full_class_name = model_config.pop('class', None) # get model_config['class'] and delete 'class' item
     module_name, class_name = re.findall("<class '([0-9a-zA-Z_\.]+)\.([0-9a-zA-Z_]+)'>", full_class_name)[0]
     class_obj = getattr(importlib.import_module(module_name), class_name)
     return class_obj(**model_config) # get a model object

def save_model_config(model_config: Dict, path: str) -> None:
    assert ('class' in model_config), "The model configuration should contain the class name"
    json.dump(model_config, open(path, 'w'), indent=4)

def save_model_state_dict(model_state_dict: Dict, path: str) -> None:
    model_state_dict_at_cpu = {k: v.cpu() for k, v in list(model_state_dict.items())}
    torch.save(model_state_dict_at_cpu, path)

def save_options(options: Dict, path: str) -> None:
    json.dump(options, open(path, 'w'), indent=4)

def save_model_with_config(model: nn.Module, model_path: str) -> None:
    """ Given the model and the path to the model, save the model ($dir/model_name.mdl)
    along with its configuration ($dir/model_name.conf) at the same time. """
    assert model_path.endswith(".mdl"), "model '{}' should end with '.mdl'".format(model_path)
    config_path = os.path.splitext(model_path)[0] + ".conf"
    save_model_config(model.get_config(), config_path)
    save_model_state_dict(model.state_dict(), model_path)

def load_pretrained_model_with_config(model_path: str) -> nn.Module:
    """ Given the path to the model, load the model ($dir/model_name.mdl)
    along with its configuration ($dir/model_name.conf) at the same time. """
    assert model_path.endswith(".mdl"), "model '{}' should end with '.mdl'".format(model_path)
    config_path = os.path.splitext(model_path)[0] + ".conf"
    model_config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    pretrained_model = load_model_config(model_config)
    pretrained_model.load_state_dict(torch.load(model_path))
    return pretrained_model

# -*- coding: utf-8 -*-
import pandas as pd
import numpy
from configs_json import *
from utils.config import  Config
from .dataloader import DataLoader 
if __name__ == "__main__":
    train = DataLoader().load_data(Config.from_json(cfg).data)
    train.head()
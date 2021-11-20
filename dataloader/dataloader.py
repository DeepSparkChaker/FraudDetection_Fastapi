# -*- coding: utf-8 -*-
import pandas as pd
from configs_json import *

"""Data Loader"""
class DataLoader:
    """Data Loader class"""
    @staticmethod
    def load_data(data_config):
        """Loads dataset from path"""
        return pd.read_csv(data_config.path)
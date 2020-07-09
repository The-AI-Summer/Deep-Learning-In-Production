# -*- coding: utf-8 -*-
"""Data Loader"""

import jsonschema

import tensorflow_datasets as tfds

from configs.data_schema import SCHEMA


class DataLoader:
    """Data Loader class"""

    @staticmethod
    def load_data(data_config):
        """Loads dataset from path"""
        return tfds.load(data_config.path, with_info=data_config.load_with_info)

    @staticmethod
    def validate_schema(data_point):
        jsonschema.validate({'image':data_point.tolist()},SCHEMA)


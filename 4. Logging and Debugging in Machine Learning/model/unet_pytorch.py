# -*- coding: utf-8 -*-
"""Unet model"""

# standard library

# internal
from .base_model import BaseModel


class UNet(BaseModel):
    """Unet Model Class"""
    def __init__(self, config):
        super().__init__(config)


    def load_data(self):
        # it loads data from tensorflow dataset. we probably should keep that an see if they can be converted to pytorch. but your choice

    def build(self):


    def train(self):


    def evaluate(self):


# -*- coding: utf-8 -*-
"""Abstract base model"""

from abc import ABC, abstractmethod

from utils.config import Config


class BaseModel(ABC):
    """Abstract Model class that is inherited to all models"""

    def __init__(self, cfg):
        self.config = Config.from_json(cfg)

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

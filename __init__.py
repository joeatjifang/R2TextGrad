import os

from .variable import Variable
from .loss import TextLoss
from .model import BlackboxLLM
from .engine import EngineLM, get_engine
from optimizer.optimizer import TextualGradientDescent
from .config import set_backward_engine, SingletonBackwardEngine

singleton_backward_engine = SingletonBackwardEngine()
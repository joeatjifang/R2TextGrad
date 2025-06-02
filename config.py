from .singleton import SingletonBackwardEngine
from engine import EngineLM

def set_backward_engine(engine: EngineLM) -> None:
    """Set the global backward engine for gradient computation"""
    singleton = SingletonBackwardEngine()
    singleton.engine = engine
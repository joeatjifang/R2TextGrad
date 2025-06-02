from typing import Optional
from engine import EngineLM

class SingletonBackwardEngine:
    _instance: Optional[EngineLM] = None
    
    def __new__(cls):
        if not hasattr(cls, '_instance'):
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @property
    def engine(self) -> Optional[EngineLM]:
        return self._instance
    
    @engine.setter 
    def engine(self, value: EngineLM):
        self._instance = value
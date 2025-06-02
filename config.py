from .engine import EngineLM

class SingletonBackwardEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SingletonBackwardEngine, cls).__new__(cls)
        return cls._instance

    @property
    def engine(self) -> EngineLM:
        return self._instance

    @engine.setter
    def engine(self, value: EngineLM):
        self._instance = value

def set_backward_engine(engine: EngineLM) -> None:
    singleton = SingletonBackwardEngine()
    singleton.engine = engine
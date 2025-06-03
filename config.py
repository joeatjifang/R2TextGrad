from .engine import EngineLM, get_engine

class SingletonBackwardEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SingletonBackwardEngine, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'engine'):
            self.engine: EngineLM = None

    @property
    def engine(self) -> EngineLM:
        return self._instance

    def set_engine(self, engine: EngineLM, override: bool = False):
        """
        Sets the backward engine.

        :param engine: The backward engine to set.
        :type engine: EngineLM
        :param override: Whether to override the existing engine if it is already set. Defaults to False.
        :type override: bool
        :raises Exception: If the engine is already set and override is False.
        :return: None
        """
        if ((self.engine is not None) and (not override)):
            raise Exception("Engine already set. Use override=True to override cautiously.")
        self.engine = engine

    def get_engine(self):
        """
        Returns the backward engine.

        :return: The backward engine.
        :rtype: EngineLM
        """
        return self.engine


def set_backward_engine(engine: EngineLM) -> None:
    singleton = SingletonBackwardEngine()
    singleton.engine = engine

def validate_engine_or_get_default(engine):
    if (engine is None) and (SingletonBackwardEngine().get_engine() is None):
        raise Exception(
            "No engine provided. Either provide an engine as the argument to this call, or use `textgrad.set_backward_engine(engine)` to set the backward engine.")
    elif engine is None:
        engine = SingletonBackwardEngine().get_engine()
    if isinstance(engine, str):
        engine = get_engine(engine)
    return engine
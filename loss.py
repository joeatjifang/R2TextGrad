from engine import EngineLM, get_engine
from variable import Variable
from typing import List, Union
from optimizer.autograd import LLMCall
from optimizer.autograd import Module
from .config import SingletonBackwardEngine


class TextLoss(Module):
    def __init__(self, 
                 eval_system_prompt: Union[Variable, str],
                 engine: Union[EngineLM, str] = None):
        """
        A vanilla loss function to evaluate a response.
        In particular, this module is used to evaluate any given text object.

        :param evaluation_instruction: The evaluation instruction variable.
        :type evaluation_instruction: Variable
        :param engine: The EngineLM object.
        :type engine: EngineLM
        
        :example:
        >>> from textgrad import get_engine, Variable
        >>> from textgrad.loss import TextLoss
        >>> engine = get_engine("gpt-4o")
        >>> evaluation_instruction = Variable("Is ths a good joke?", requires_grad=False)
        >>> response_evaluator = TextLoss(evaluation_instruction, engine)
        >>> response = Variable("What did the fish say when it hit the wall? Dam.", requires_grad=True)
        >>> response_evaluator(response)
        """
        super().__init__()
        if isinstance(eval_system_prompt, str):
            eval_system_prompt = Variable(eval_system_prompt, requires_grad=False, role_description="system prompt for the evaluation")
        self.eval_system_prompt = eval_system_prompt
        if ((engine is None) and (SingletonBackwardEngine().get_engine() is None)):
            raise Exception("No engine provided. Either provide an engine as the argument to this call, or use `textgrad.set_backward_engine(engine)` to set the backward engine.")
        elif engine is None:
            engine = SingletonBackwardEngine().get_engine()
        if isinstance(engine, str):
            engine = get_engine(engine)
        self.engine = engine
        self.llm_call = LLMCall(self.engine, self.eval_system_prompt)

    def forward(self, instance: Variable):
        """
        Calls the ResponseEvaluation object.

        :param instance: The instance variable.
        :type instance: Variable
        :return: The result of the evaluation
        """
        return self.llm_call(instance)

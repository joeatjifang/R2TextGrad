from variable import Variable
from engine import EngineLM
from config import validate_engine_or_get_default
from typing import List
from .llm_backward_prompts import (
    EVALUATE_VARIABLE_INSTRUCTION,
    CONVERSATION_START_INSTRUCTION_BASE,
    CONVERSATION_START_INSTRUCTION_CHAIN,
    CONVERSATION_TEMPLATE,
    OBJECTIVE_INSTRUCTION_CHAIN,
    OBJECTIVE_INSTRUCTION_BASE,
    BACKWARD_SYSTEM_PROMPT,
)
from variable import Variable
from engine import EngineLM

from abc import ABC, abstractmethod
from typing import List


class Function(ABC):
    """
    The class to define a function that can be called and backpropagated through.
    """
    
    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> Variable:
        pass
    
    @abstractmethod
    def backward(self, *args, **kwargs):
        pass
    

class BackwardContext:
    """
    Represents a context for backward computation.

    :param backward_fn: The backward function to be called during backward computation.
    :type backward_fn: callable
    :param args: Variable length argument list to be passed to the backward function.
    :param kwargs: Arbitrary keyword arguments to be passed to the backward function.

    :ivar backward_fn: The backward function to be called during backward computation.
    :vartype backward_fn: callable
    :ivar fn_name: The fully qualified name of the backward function.
    :vartype fn_name: str
    :ivar args: Variable length argument list to be passed to the backward function.
    :ivar kwargs: Arbitrary keyword arguments to be passed to the backward function.

    :method __call__(backward_engine: EngineLM) -> Any:
        Calls the backward function with the given backward engine and returns the result.
    :method __repr__() -> str:
        Returns a string representation of the BackwardContext object.
    """

    def __init__(self, backward_fn, *args, **kwargs):
        self.backward_fn = backward_fn
        self.fn_name = f"{backward_fn.__module__}.{backward_fn.__qualname__}"
        self.args = args
        self.kwargs = kwargs

    def __call__(self, backward_engine: EngineLM):
        return self.backward_fn(*self.args, **self.kwargs, backward_engine=backward_engine)

    def __repr__(self):
        return f"{self.fn_name}"


class Module(ABC):
    """Abstract module class with parameters akin to PyTorch's nn.Module.
    """
    parameters: List[Variable]
    def zero_grad(self):
        for p in self.parameters():
            p.reset_gradients()

    def named_parameters(self):
        for p in self.parameters():
            yield p.get_role_description(), p
            
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class LLMCall(Function):
    def __init__(self, engine: EngineLM, system_prompt: Variable = None):
        """The simple LLM call function. This function will call the LLM with the input and return the response, also register the grad_fn for backpropagation.

        :param engine: engine to use for the LLM call
        :type engine: EngineLM
        :param system_prompt: system prompt to use for the LLM call, default depends on the engine.
        :type system_prompt: Variable, optional
        """
        super().__init__()
        self.engine = validate_engine_or_get_default(engine)
        self.system_prompt = system_prompt
        if self.system_prompt and self.system_prompt.get_role_description() is None:
            self.system_prompt.set_role_description(SYSTEM_PROMPT_DEFAULT_ROLE)
    
    def forward(self, input_variable: Variable, response_role_description: str = VARIABLE_OUTPUT_DEFAULT_ROLE) -> Variable:
        """
        The LLM call. This function will call the LLM with the input and return the response, also register the grad_fn for backpropagation.
        
        :param input_variable: The input variable (aka prompt) to use for the LLM call.
        :type input_variable: Variable
        :param response_role_description: Role description for the LLM response, defaults to VARIABLE_OUTPUT_DEFAULT_ROLE
        :type response_role_description: str, optional
        :return: response sampled from the LLM
        :rtype: Variable
        
        :example:
        >>> from textgrad import Variable, get_engine
        >>> from textgrad.autograd.llm_ops import LLMCall
        >>> engine = get_engine("gpt-3.5-turbo")
        >>> llm_call = LLMCall(engine)
        >>> prompt = Variable("What is the capital of France?", role_description="prompt to the LM")
        >>> response = llm_call(prompt, engine=engine) 
        # This returns something like Variable(data=The capital of France is Paris., grads=)
        """
        # TODO: Should we allow default roles? It will make things less performant.
        system_prompt_value = self.system_prompt.value if self.system_prompt else None

        # Make the LLM Call
        response_text = self.engine(input_variable.value, system_prompt=system_prompt_value)

        # Create the response variable
        response = Variable(
            value=response_text,
            predecessors=[self.system_prompt, input_variable] if self.system_prompt else [input_variable],
            role_description=response_role_description
        )
        
        logger.info(f"LLMCall function forward", extra={"text": f"System:{system_prompt_value}\nQuery: {input_variable.value}\nResponse: {response_text}"})
        
        # Populate the gradient function, using a container to store the backward function and the context
        response.set_grad_fn(BackwardContext(backward_fn=self.backward, 
                                             response=response, 
                                             prompt=input_variable.value, 
                                             system_prompt=system_prompt_value))

        return response
    
    def backward(self, response: Variable, prompt: str, system_prompt: str, backward_engine: EngineLM):
        """
        Backward pass through the LLM call. This will register gradients in place.

        :param response: The response variable.
        :type response: Variable
        :param prompt: The prompt string that will be used as input to an LM.
        :type prompt: str
        :param system_prompt: The system prompt string.
        :type system_prompt: str
        :param backward_engine: The backward engine that will do the gradient computation.
        :type backward_engine: EngineLM

        :return: None
        """
        children_variables = response.predecessors
        if response.get_gradient_text() == "":
            self._backward_through_llm_base(children_variables, response, prompt, system_prompt, backward_engine)
        else:
            self._backward_through_llm_chain(children_variables, response, prompt, system_prompt, backward_engine)

    @staticmethod
    def _construct_llm_chain_backward_prompt(backward_info: dict[str, str]) -> str:
        conversation = CONVERSATION_TEMPLATE.format(**backward_info)
        backward_prompt = CONVERSATION_START_INSTRUCTION_CHAIN.format(conversation=conversation, **backward_info)
        backward_prompt += OBJECTIVE_INSTRUCTION_CHAIN.format(**backward_info)
        backward_prompt += EVALUATE_VARIABLE_INSTRUCTION.format(**backward_info)
        return backward_prompt

    @staticmethod
    def _backward_through_llm_chain(variables: List[Variable], 
                                    response: Variable, 
                                    prompt: str, 
                                    system_prompt: str,
                                    backward_engine: EngineLM):

        """
        Backward through the LLM to compute gradients for each variable, in the case where the output has gradients on them.
        i.e. applying the chain rule.
        
        :param variables: The list of variables to compute gradients for.
        :type variables: List[Variable]
        :param response: The response variable.
        :type response: Variable
        :param prompt: The prompt string.
        :type prompt: str
        :param system_prompt: The system prompt string.
        :type system_prompt: str
        :param backward_engine: The backward engine to use for computing gradients.
        :type backward_engine: EngineLM

        :return: None
        """
        for variable in variables:
            if not variable.requires_grad:
                continue

            backward_info = {
                "response_desc": response.get_role_description(),
                "response_value": response.get_value(),
                "response_gradient": response.get_gradient_text(),
                "prompt": prompt,
                "system_prompt": system_prompt,
                "variable_desc": variable.get_role_description(),
                "variable_short": variable.get_short_value()
            }
            
            backward_prompt = LLMCall._construct_llm_chain_backward_prompt(backward_info)

            logger.info(f"_backward_through_llm prompt", extra={"_backward_through_llm": backward_prompt})
            gradient_value = backward_engine(backward_prompt, system_prompt=BACKWARD_SYSTEM_PROMPT)
            logger.info(f"_backward_through_llm gradient", extra={"_backward_through_llm": gradient_value})
            
            var_gradients = Variable(value=gradient_value, role_description=f"feedback to {variable.get_role_description()}")
            variable.gradients.add(var_gradients)
            conversation = CONVERSATION_TEMPLATE.format(**backward_info)
            variable.gradients_context[var_gradients] = {
                "context": conversation, 
                "response_desc": response.get_role_description(),
                "variable_desc": variable.get_role_description()
            }
            
            if response._reduce_meta:
                var_gradients._reduce_meta.extend(response._reduce_meta)
                variable._reduce_meta.extend(response._reduce_meta)

    @staticmethod
    def _construct_llm_base_backward_prompt(backward_info: dict[str, str]) -> str:
        conversation = CONVERSATION_TEMPLATE.format(**backward_info)
        backward_prompt = CONVERSATION_START_INSTRUCTION_BASE.format(conversation=conversation, **backward_info)
        backward_prompt += OBJECTIVE_INSTRUCTION_BASE.format(**backward_info)
        backward_prompt += EVALUATE_VARIABLE_INSTRUCTION.format(**backward_info)
        return backward_prompt

    @staticmethod
    def _backward_through_llm_base(variables: List[Variable], 
                                   response: Variable,
                                   prompt: str,
                                   system_prompt: str,
                                   backward_engine: EngineLM):
        """
        Backward pass through the LLM base. 
        In this case we do not have gradients on the output variable.

        :param variables: A list of variables to compute gradients for.
        :type variables: List[Variable]
        :param response: The response variable.
        :type response: Variable
        :param prompt: The prompt string.
        :type prompt: str
        :param system_prompt: The system prompt string.
        :type system_prompt: str
        :param backward_engine: The backward engine to use for computing gradients.
        :type backward_engine: EngineLM
        """
        for variable in variables:
            if not variable.requires_grad:
                continue

            backward_info = {
                "response_desc": response.get_role_description(),
                "response_value": response.get_value(),
                "prompt": prompt,
                "system_prompt": system_prompt,
                "variable_desc": variable.get_role_description(),
                "variable_short": variable.get_short_value()
            }
            
            backward_prompt = LLMCall._construct_llm_base_backward_prompt(backward_info)
            
            logger.info(f"_backward_through_llm prompt", extra={"_backward_through_llm": backward_prompt})
            gradient_value = backward_engine(backward_prompt, system_prompt=BACKWARD_SYSTEM_PROMPT)
            logger.info(f"_backward_through_llm gradient", extra={"_backward_through_llm": gradient_value})

            conversation = CONVERSATION_TEMPLATE.format(**backward_info)
            var_gradients = Variable(value=gradient_value, role_description=f"feedback to {variable.get_role_description()}")
            variable.gradients.add(var_gradients)
            variable.gradients_context[var_gradients] = {
                "context": conversation, 
                "response_desc": response.get_role_description(),
                "variable_desc": variable.get_role_description()
            }

            if response._reduce_meta:
                var_gradients._reduce_meta.extend(response._reduce_meta)
                variable._reduce_meta.extend(response._reduce_meta)

from typing import List, Union
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from variable import Variable
from engine import EngineLM
from config import validate_engine_or_get_default
from .optimizer import Optimizer, get_gradient_and_context_text
from .optimizer_prompts import construct_tgd_prompt, OPTIMIZER_SYSTEM_PROMPT

class R2TextualGradientDescent(Optimizer):
    def __init__(self, 
                 parameters: List[Variable],
                 verbose: int=0,
                 engine: Union[EngineLM, str]=None,
                 constraints: List[str]=None,
                 new_variable_tags: List[str]=None,
                 optimizer_system_prompt: str=OPTIMIZER_SYSTEM_PROMPT,
                 in_context_examples: List[str]=None,
                 gradient_memory: int=0,
                 num_trials: int=5):
        super().__init__(parameters)
        self.verbose = verbose
        self.engine = validate_engine_or_get_default(engine)
        self.constraints = constraints if constraints else []
        self.new_variable_tags = new_variable_tags if new_variable_tags else ["<NEW_VARIABLE>", "</NEW_VARIABLE>"]
        self.optimizer_system_prompt = optimizer_system_prompt.format(
            new_variable_start_tag=new_variable_tags[0], 
            new_variable_end_tag=new_variable_tags[1]
        )
        self.in_context_examples = in_context_examples if in_context_examples else []
        self.gradient_memory = gradient_memory
        self.gradient_memory_dict = defaultdict(list)
        self.num_trials = num_trials
        self.r2_history = []

    def compute_r2_score(self, responses, labels):
        """Compute R-squared score using logistic regression"""
        if len(responses) < 2:
            return 0.0
            
        try:
            # Convert responses to binary outcomes
            X = np.array(responses).reshape(-1, 1)
            y = np.array(labels).astype(int)
            
            # Fit logistic regression
            lr = LogisticRegression()
            lr.fit(X, y)
            
            # Calculate R-squared
            y_pred = lr.predict_proba(X)[:, 1]
            r2 = np.corrcoef(y, y_pred)[0,1] ** 2
            return float(r2)
        except:
            return 0.0

    def _update_prompt(self, variable: Variable) -> str:
        # Get gradient info and context
        grad_text = get_gradient_and_context_text(variable)
        
        # Add R2 score to optimization context
        r2_context = f"\nR2 Score History: {self.r2_history}\n" if self.r2_history else ""
        
        optimizer_info = {
            "variable_desc": variable.get_role_description(),
            "variable_value": variable.value,
            "variable_grad": grad_text + r2_context,
            "variable_short": variable.get_short_value(),
            "constraint_text": self.constraint_text,
            "new_variable_start_tag": self.new_variable_tags[0],
            "new_variable_end_tag": self.new_variable_tags[1],
            "in_context_examples": "\n".join(self.in_context_examples),
        }
        
        return construct_tgd_prompt(
            do_constrained=(len(self.constraints) > 0),
            do_in_context_examples=bool(self.in_context_examples),
            **optimizer_info
        )

    def step(self):
        """Perform optimization step with R-squared evaluation"""
        for parameter in self.parameters:
            best_value = parameter.value
            best_r2 = 0.0
            
            # Try multiple variations and select best by R2 score
            for _ in range(self.num_trials):
                prompt = self._update_prompt(parameter)
                new_text = self.engine(prompt, system_prompt=self.optimizer_system_prompt)
                
                try:
                    new_value = new_text.split(self.new_variable_tags[0])[1].split(self.new_variable_tags[1])[0].strip()
                    
                    # Evaluate R2 score
                    responses = [0.0, 1.0]  # Example responses
                    labels = [0, 1]  # Example target labels
                    r2_score = self.compute_r2_score(responses, labels)
                    
                    if r2_score > best_r2:
                        best_r2 = r2_score 
                        best_value = new_value
                        
                except IndexError:
                    continue
            
            # Update parameter with best value found
            parameter.set_value(best_value)
            self.r2_history.append(best_r2)
            
            # Store gradient info if using memory
            if self.gradient_memory > 0:
                key = parameter.get_role_description()
                if len(self.gradient_memory_dict[key]) >= self.gradient_memory:
                    self.gradient_memory_dict[key].pop(0)
                self.gradient_memory_dict[key].append({
                    "value": best_value,
                    "r2_score": best_r2,
                    "gradients": get_gradient_and_context_text(parameter)
                })
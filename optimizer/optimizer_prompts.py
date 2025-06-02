GLOSSARY_TEXT = """
### Glossary of tags that will be sent to you:
# - <LM_SYSTEM_PROMPT>: The system prompt for the language model.
# - <LM_INPUT>: The input to the language model.
# - <LM_OUTPUT>: The output of the language model.
# - <FEEDBACK>: The feedback to the variable.
# - <CONVERSATION>: The conversation history.
# - <FOCUS>: The focus of the optimization.
# - <ROLE>: The role description of the variable."""

### Optimize Prompts

# System prompt to TGD
OPTIMIZER_SYSTEM_PROMPT = """
Your task is to optimize the prompt using R-squared scores from logistic regression.
The optimization should maximize the R-squared value while maintaining semantic meaning.
Use tags {new_variable_start_tag} and {new_variable_end_tag} for new variables.
"""

def construct_tgd_prompt(do_constrained: bool = False,
                        do_in_context_examples: bool = False,
                        **kwargs) -> str:
    """Construct the prompt template for R-squared optimization"""
    template = f"""
    Current R2 Score: {kwargs.get('variable_grad', '')}
    Role Description: {kwargs.get('variable_desc', '')}
    Current Value: {kwargs.get('variable_value', '')}
    
    Optimize the above prompt to maximize R2 score while maintaining its purpose.
    Provide your optimized version between {kwargs.get('new_variable_start_tag')} and {kwargs.get('new_variable_end_tag')}.
    """
    return template

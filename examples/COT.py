from optimizer import R2TextualGradientDescent
from variable import Variable
from engine.deepseek import ChatDeepSeek # You can choose deepseek or llama

# Example patent text and product name
PATENT_TEXT = """
A method and system for improving battery life in mobile devices through 
dynamic power management and selective component activation based on 
usage patterns and sensor data.
"""

PRODUCT_NAME = "SmartPhone Power Optimizer"

def create_cot_prompt(product_name: str, patent_text: str) -> str:
    return f"""
    Let's evaluate the relevance of this patent to the described product step by step.
    Score from 1 (no relevance) to 10 (direct and critical applicability).

    Product: {product_name}
    Patent: {patent_text}

    Let's think about this systematically:

    1. First, identify the key technical components of the patent:
    - What are the main technological elements?
    - What problem does it solve?
    - What methods does it use?

    2. Then, analyze the product's likely features:
    - What capabilities would this product need?
    - What technical approaches would it use?
    - What problems does it aim to solve?

    3. Compare patent and product:
    - Where do they overlap technically?
    - Are their goals aligned?
    - Would the patent's methods be essential for the product?

    4. Final Evaluation:
    Based on the above analysis, provide a relevance score (1-10) with specific 
    reasoning for why this score was chosen.

    Follow this reasoning structure carefully to arrive at your conclusion.
    """

def main():
    # Initialize DeepSeek engine
    engine = ChatDeepSeek(
        model_string="deepseek-v3",
        temperature=0.7
    )

    # Initialize the base prompt
    initial_prompt = create_cot_prompt(PRODUCT_NAME, PATENT_TEXT)

    # Create variable for optimization
    prompt_var = Variable(
        value=initial_prompt,
        role_description="chain-of-thought patent-product relevance evaluation prompt"
    )

    # Initialize optimizer with DeepSeek engine
    optimizer = R2TextualGradientDescent(
        parameters=[prompt_var],
        engine=engine,  # Using DeepSeek instead of ChatAnthropic
        num_trials=5,
        gradient_memory=3
    )

    # Define constraints for the prompt
    constraints = [
        "Must maintain numerical scoring from 1-10",
        "Must preserve step-by-step reasoning structure",
        "Must include all analysis steps",
        "Must focus on technical comparison",
        "Must justify final score with specific reasoning"
    ]

    # Run optimization
    NUM_ITERATIONS = 3
    print("Starting COT prompt optimization...")
    print(f"Initial prompt:\n{prompt_var.value}\n")

    for i in range(NUM_ITERATIONS):
        print(f"Iteration {i+1}/{NUM_ITERATIONS}")
        optimizer.step()
        print(f"Current R2 score: {optimizer.r2_history[-1]:.4f}")
        print(f"Updated prompt:\n{prompt_var.value}\n")

    print("Optimization complete!")
    print(f"Final R2 score: {optimizer.r2_history[-1]:.4f}")
    print(f"Final optimized prompt:\n{prompt_var.value}")

if __name__ == "__main__":
    main()
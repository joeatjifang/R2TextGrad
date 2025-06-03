from optimizer import R2TextualGradientDescent
from variable import Variable
from engine.deepseek import ChatDeepSeek

# Example patent text and product name
PATENT_TEXT = """
A method and system for improving battery life in mobile devices through 
dynamic power management and selective component activation based on 
usage patterns and sensor data.
"""

PRODUCT_NAME = "SmartPhone Power Optimizer"

def create_relevance_prompt(product_name: str, patent_text: str) -> str:
    return f"""
    In your knowledge of this industry, evaluating the relevance of the patent 
    to the described product. Remember that score from 1 (no relevance) to 10 
    (direct and critical applicatility)

    Product: {product_name}
    Patent: {patent_text}
    """

def main():
    # Initialize DeepSeek engine
    engine = ChatDeepSeek(
        model_string="deepseek-v3",
        temperature=0.7
    )

    # Initialize the base prompt
    initial_prompt = create_relevance_prompt(PRODUCT_NAME, PATENT_TEXT)

    # Create variable for optimization
    prompt_var = Variable(
        value=initial_prompt,
        role_description="patent-product relevance evaluation prompt"
    )

    # Initialize optimizer
    optimizer = R2TextualGradientDescent(
        parameters=[prompt_var],
        engine=engine,
        num_trials=5,
        gradient_memory=3
    )

    # Define constraints for the prompt
    constraints = [
        "Must maintain numerical scoring from 1-10",
        "Must focus on relevance evaluation",
        "Must preserve product and patent context"
    ]

    # Run optimization
    NUM_ITERATIONS = 3
    print("Starting prompt optimization...")
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
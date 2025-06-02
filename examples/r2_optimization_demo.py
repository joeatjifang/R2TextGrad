from optimizer import R2TextualGradientDescent
from variable import Variable
from engine.anthropic import ChatAnthropic  # Assuming this exists

def main():
    # Initialize the prompt to be optimized
    initial_prompt = """
    Given a text, analyze its sentiment and classify it as positive or negative.
    Provide your answer as 'positive' or 'negative' only.
    """

    # Create variable
    prompt_var = Variable(
        value=initial_prompt,
        role_description="Sentiment analysis prompt optimization"
    )

    # Initialize optimizer with Claude engine
    optimizer = R2TextualGradientDescent(
        parameters=[prompt_var],
        engine=ChatAnthropic(),
        num_trials=5,
        gradient_memory=3
    )

    # Example constraints
    constraints = [
        "Must maintain focus on sentiment analysis",
        "Output format must remain 'positive' or 'negative'"
    ]

    # Run optimization steps
    NUM_ITERATIONS = 3
    print("Starting prompt optimization...")
    print(f"Initial prompt: {prompt_var.value}\n")

    for i in range(NUM_ITERATIONS):
        print(f"Iteration {i+1}/{NUM_ITERATIONS}")
        optimizer.step()
        print(f"Current R2 score: {optimizer.r2_history[-1]:.4f}")
        print(f"Updated prompt: {prompt_var.value}\n")

    print("Optimization complete!")
    print(f"Final R2 score: {optimizer.r2_history[-1]:.4f}")
    print(f"Final prompt: {prompt_var.value}")

if __name__ == "__main__":
    main()

from optimizer import R2TextualGradientDescent
from variable import Variable
from engine.deepseek import ChatDeepSeek
from typing import List
import sys
import os

# Add the main directory to system path to import RAG
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RAG import RAGRetriever

# Example patent text and product name
PATENT_TEXT = """
A method and system for improving battery life in mobile devices through 
dynamic power management and selective component activation based on 
usage patterns and sensor data.
"""

PRODUCT_NAME = "SmartPhone Power Optimizer"

# Example product database (in practice, this would be loaded from a real database)
PRODUCT_DATABASE = [
    """The SmartPhone Power Optimizer utilizes advanced AI algorithms to 
    dynamically manage power consumption across all device components.""",
    
    """Our power management solution continuously monitors device usage patterns
    and adjusts power allocation in real-time.""",
    
    """Intelligent sensor data processing allows for optimal battery performance
    while maintaining full device functionality.""",
    
    """The system includes automated background app management and custom
    power profiles for different usage scenarios."""
]

def summarize_product_features(descriptions: List[str], engine: ChatDeepSeek) -> str:
    """
    Use LLM to summarize the retrieved product descriptions into a concise format
    """
    summarization_prompt = f"""
    Summarize these product feature descriptions into a single coherent paragraph:

    {chr(10).join(f'- {desc.strip()}' for desc in descriptions)}

    Provide a clear and concise summary that captures all key technological aspects.
    """
    
    response = engine.generate(summarization_prompt)
    return response.strip()

def create_relevance_prompt(product_name: str, product_summary: str, patent_text: str) -> str:
    return f"""
    Based on the following product description and patent text, evaluate their relevance.
    Score from 1 (no relevance) to 10 (direct and critical applicability).

    Product Name: {product_name}
    
    Product Description:
    {product_summary}
    
    Patent Description:
    {patent_text}

    Provide your evaluation with detailed reasoning based on the specific matches 
    between product features and patent claims.
    """

def main():
    # Initialize DeepSeek engine
    engine = ChatDeepSeek(
        model_string="deepseek-v3",
        temperature=0.7
    )
    retriever = RAGRetriever(documents=PRODUCT_DATABASE)
    
    # Create combined query for similarity search
    search_query = f"{PRODUCT_NAME} {PATENT_TEXT}"
    
    # Retrieve top 3 most relevant product descriptions
    relevant_descriptions = retriever.get_relevant_passages(
        query=search_query,
        k=3
    )

    # Summarize retrieved descriptions using LLM
    PRODUCT_DESCRIPTION = summarize_product_features(relevant_descriptions, engine)

    # Initialize the base prompt with summarized description
    initial_prompt = create_relevance_prompt(
        PRODUCT_NAME, 
        PRODUCT_DESCRIPTION, 
        PATENT_TEXT
    )

    # Create variable for optimization
    prompt_var = Variable(
        value=initial_prompt,
        role_description="RAG-enhanced patent-product relevance evaluation prompt"
    )

    # Initialize optimizer with the same engine instance
    optimizer = R2TextualGradientDescent(
        parameters=[prompt_var],
        engine=engine,  # Use the same engine instance
        num_trials=5,
        gradient_memory=3
    )

    # Define constraints for the prompt
    constraints = [
        "Must maintain numerical scoring from 1-10",
        "Must focus on relevance evaluation",
        "Must preserve product and patent context",
        "Must utilize summarized product description",
        "Must consider specific feature matches"
    ]

    # Run optimization
    NUM_ITERATIONS = 3
    print("Starting RAG-enhanced prompt optimization...")
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
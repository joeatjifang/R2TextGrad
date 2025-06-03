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

# Example product database
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
    with key technological aspects preserved
    """
    summarization_prompt = f"""
    Synthesize these product feature descriptions into a comprehensive technical summary:

    {chr(10).join(f'- {desc.strip()}' for desc in descriptions)}

    Focus on:
    1. Core technological capabilities
    2. Key implementation methods
    3. Distinctive features
    
    Provide a clear, structured summary that maintains all critical technical details.
    """
    
    response = engine.generate(summarization_prompt)
    return response.strip()

def create_cot_relevance_prompt(product_name: str, product_summary: str, patent_text: str) -> str:
    return f"""
    Let's evaluate the relevance between this patent and product through systematic analysis.
    Score from 1 (no relevance) to 10 (direct and critical applicability).

    Product Name: {product_name}
    
    Product Technical Summary:
    {product_summary}
    
    Patent Description:
    {patent_text}

    Let's analyze step by step:

    1. Core Technology Analysis:
    - What are the key technological components in the patent?
    - What specific features does the product implement?
    - Identify the technical overlaps between them.

    2. Implementation Method Comparison:
    - How does the patent suggest implementing its solution?
    - How does the product actually implement its features?
    - Analyze the similarity in approaches.

    3. Problem-Solution Alignment:
    - What problem does the patent aim to solve?
    - What problem does the product address?
    - Evaluate how well these align.

    4. Feature-Claim Matching:
    - Match each product feature with relevant patent claims
    - Identify which features directly implement patent methods
    - Note any features that go beyond or differ from the patent

    5. Final Evaluation:
    Based on the above systematic analysis:
    - Provide a relevance score (1-10)
    - Justify the score with specific references to the matches identified
    - Highlight the most critical overlapping aspects

    Follow this reasoning structure to reach a well-justified conclusion.
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
    initial_prompt = create_cot_relevance_prompt(
        PRODUCT_NAME, 
        PRODUCT_DESCRIPTION, 
        PATENT_TEXT
    )

    # Create variable for optimization
    prompt_var = Variable(
        value=initial_prompt,
        role_description="RAG-enhanced COT patent-product relevance evaluation prompt"
    )

    # Initialize optimizer with the same engine instance
    optimizer = R2TextualGradientDescent(
        parameters=[prompt_var],
        engine=engine,  # Reuse the same engine instance
        num_trials=5,
        gradient_memory=3
    )

    # Define constraints for the prompt
    constraints = [
        "Must maintain numerical scoring from 1-10",
        "Must follow the step-by-step analysis structure",
        "Must preserve product and patent context",
        "Must utilize summarized product description",
        "Must justify score with specific feature matches",
        "Must maintain chain-of-thought reasoning steps",
        "Must reference specific technical details from the summary"
    ]

    # Run optimization
    NUM_ITERATIONS = 3
    print("Starting RAG-enhanced COT prompt optimization...")
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
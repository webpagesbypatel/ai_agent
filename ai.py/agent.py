from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel
from typing import TypedDict, Literal
import logging

logger = logging.getLogger(__name__)

rag_prompt = PromptTemplate(
    template="""You are an expert assistant who answers questions concisely based ONLY on the provided context.

    Use the following retrieved context to answer the user's question.
    - Do not use any information outside of this context.
    - If the context does not contain the answer, you must respond with "I do not have enough information to answer this question."
    - Keep your answer to a maximum of three sentences.

    Question: {question}

    Context: {context}

    Answer:""",
    input_variables=["question", "context"],
)

reflection_prompt = PromptTemplate(
    template="""You are a meticulous quality reviewer. Your task is to evaluate a generated answer based on its relevance to the question and its faithfulness to the provided context.

    Review the following items:

    1. Original Question

    2. Retrieved Context

    3. Generated Answer

    Does the Generated Answer directly respond to the Original Question?

    Is the Generated Answer fully supported by the information in the Retrieved Context?

    If both conditions are met, respond with the single phrase: 'The answer is relevant and complete.'

    Otherwise, provide a concise critique of why the answer is insufficient or unfaithful to the context.

    Question: {question}

    Context: {context}

    Generated Answer: {generation}

    Your Review:""",
    input_variables=["question", "context", "generation"],
)


# State definition for the self-correction loop
class GraphState(TypedDict):
    question: str
    context: str
    generation: str
    reflection: str
    iteration: int
    max_iterations: int  # Store in state for routing function access
    reflection_history: list[str]  # Track all reflections for debugging


def reflect_node(state: GraphState, config: dict) -> GraphState:
    """
    Reflection node that evaluates the generated answer.
    Uses the reflection_prompt to get an evaluation from the LLM.
    
    Args:
        state: Current graph state
        config: LangGraph config containing 'llm' key
        
    Returns:
        Updated state with reflection and incremented iteration
    """
    try:
        # Get LLM from config (LangGraph pattern)
        llm = config.get("llm")
        if not llm:
            raise ValueError("LLM not found in config")
        
        # Format the reflection prompt with current state
        formatted_prompt = reflection_prompt.format(
            question=state["question"],
            context=state["context"],
            generation=state["generation"]
        )
        
        # Get evaluation from LLM
        response = llm.invoke(formatted_prompt)
        reflection_text = response.content if hasattr(response, 'content') else str(response)
        
        # Increment iteration counter
        current_iteration = state.get("iteration", 0)
        
        # Get reflection history (initialize if not present)
        reflection_history = state.get("reflection_history", [])
        reflection_history.append(reflection_text)
        
        # Return new state (immutable pattern)
        return {
            **state,
            "reflection": reflection_text,
            "iteration": current_iteration + 1,
            "reflection_history": reflection_history
        }
        
    except Exception as e:
        logger.error(f"Error in reflect_node: {e}")
        # Fallback: mark as needing regeneration if reflection fails
        return {
            **state,
            "reflection": f"Reflection failed: {str(e)}. Regenerating...",
            "iteration": state.get("iteration", 0) + 1
        }


def should_regenerate(state: GraphState) -> Literal["regenerate", "end"]:
    """
    Routing function that determines whether to regenerate or end execution.
    Checks if the reflection indicates the answer is complete, or if max iterations reached.
    
    Args:
        state: Current graph state
        
    Returns:
        "end" if answer is complete or max iterations reached, "regenerate" otherwise
    """
    # Get current iteration and max_iterations from state
    current_iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)  # Default to 3 if not set
    
    # Check if max iterations reached
    if current_iteration >= max_iterations:
        logger.warning(f"Max iterations ({max_iterations}) reached. Ending execution.")
        return "end"
    
    # Get reflection text (handle None/empty cases)
    reflection = state.get("reflection", "")
    if not reflection:
        logger.warning("Empty reflection found. Regenerating...")
        return "regenerate"
    
    # Normalize reflection for robust matching
    reflection_lower = reflection.lower().strip()
    
    # Multiple patterns to detect completion (more robust)
    completion_patterns = [
        "the answer is relevant and complete",
        "answer is relevant and complete",
        "relevant and complete",
        "answer is complete",
        "sufficient and accurate",
        "meets both conditions"
    ]
    
    # Check if any completion pattern is found
    is_complete = any(pattern in reflection_lower for pattern in completion_patterns)
    
    if is_complete:
        logger.info(f"Answer validated as complete after {current_iteration} iteration(s).")
        return "end"
    
    # Log why we're regenerating
    logger.info(f"Iteration {current_iteration}: Regenerating. Reflection: {reflection[:100]}...")
    return "regenerate"


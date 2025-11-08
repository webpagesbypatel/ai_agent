import streamlit as st
import logging
from langgraph.graph import StateGraph, END
from langchain_huggingface import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from agent import (
    GraphState, 
    rag_prompt, 
    reflection_prompt, 
    should_regenerate
)
from app import plan_prompt
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="RAG with Self-Correction",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None


def initialize_llm():
    """Initialize HuggingFace LLM"""
    if st.session_state.llm is None:
        with st.spinner("Loading HuggingFace model..."):
            try:
                # Using a smaller model for faster loading
                model_id = "microsoft/DialoGPT-medium"  # You can change this
                st.session_state.llm = HuggingFacePipeline.from_model_id(
                    model_id=model_id,
                    task="text-generation",
                    model_kwargs={"temperature": 0.7, "max_length": 512},
                )
                st.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {e}")
                st.info("Using a fallback - please install transformers and torch")
                return None
    return st.session_state.llm


def initialize_embeddings():
    """Initialize HuggingFace embeddings"""
    if st.session_state.embeddings is None:
        with st.spinner("Loading embeddings model..."):
            try:
                st.session_state.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
            except Exception as e:
                st.error(f"Error loading embeddings: {e}")
                return None
    return st.session_state.embeddings


def create_vectorstore(texts):
    """Create ChromaDB vectorstore from texts"""
    if not texts:
        return None
    
    try:
        # Split texts into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        documents = [Document(page_content=text) for text in texts]
        splits = text_splitter.split_documents(documents)
        
        # Create embeddings
        embeddings = initialize_embeddings()
        if not embeddings:
            return None
        
        # Create ChromaDB vectorstore
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        st.session_state.vectorstore = vectorstore
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vectorstore: {e}")
        return None


def retrieve_context(question: str, vectorstore, k: int = 3) -> str:
    """Retrieve relevant context from vectorstore"""
    if vectorstore is None:
        return ""
    
    try:
        docs = vectorstore.similarity_search(question, k=k)
        context = "\n\n".join([doc.page_content for doc in docs])
        return context
    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        return ""


def create_answer_node(llm):
    """Create answer node with LLM bound"""
    def answer_node(state: GraphState) -> GraphState:
        """Generate answer using RAG prompt"""
        try:
            # Format RAG prompt
            formatted_prompt = rag_prompt.format(
                question=state["question"],
                context=state["context"]
            )
            
            # Generate answer
            response = llm.invoke(formatted_prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            return {
                **state,
                "generation": answer
            }
        except Exception as e:
            logger.error(f"Error in answer_node: {e}")
            return {
                **state,
                "generation": f"Error generating answer: {str(e)}"
            }
    return answer_node


def create_reflect_node(llm):
    """Create reflect node with LLM bound"""
    def reflect_node(state: GraphState) -> GraphState:
        """Reflection node that evaluates the generated answer"""
        try:
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
    return reflect_node


def build_graph(llm):
    """Build LangGraph workflow with LLM bound to nodes"""
    workflow = StateGraph(GraphState)
    
    # Create nodes with LLM bound
    answer_node_func = create_answer_node(llm)
    reflect_node_func = create_reflect_node(llm)
    
    # Add nodes
    workflow.add_node("answer", answer_node_func)
    workflow.add_node("reflect", reflect_node_func)
    
    # Set entry point
    workflow.set_entry_point("answer")
    
    # Add edges
    workflow.add_edge("answer", "reflect")
    workflow.add_conditional_edges(
        "reflect",
        should_regenerate,
        {
            "regenerate": "answer",
            "end": END
        }
    )
    
    return workflow.compile()


# Streamlit UI
st.title("🤖 RAG with Self-Correction Loop")
st.markdown("Ask questions and get answers with automatic quality checking and self-correction")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Knowledge base setup
    st.subheader("Knowledge Base")
    knowledge_text = st.text_area(
        "Enter your knowledge base text:",
        height=200,
        placeholder="Paste your documents or knowledge base content here..."
    )
    
    if st.button("Create Knowledge Base"):
        if knowledge_text:
            texts = [knowledge_text] if knowledge_text else []
            vectorstore = create_vectorstore(texts)
            if vectorstore:
                st.success("Knowledge base created!")
        else:
            st.warning("Please enter some text first")
    
    # Model settings
    st.subheader("Settings")
    max_iterations = st.slider("Max Iterations", 1, 5, 3)
    
    if st.button("Initialize Models"):
        initialize_llm()
        initialize_embeddings()

# Main chat interface
if st.session_state.vectorstore is None:
    st.info("👈 Please create a knowledge base in the sidebar first")
else:
    # Chat input
    question = st.chat_input("Ask a question about your knowledge base...")
    
    if question:
        # Initialize LLM
        llm = initialize_llm()
        if not llm:
            st.error("Please initialize the model first")
            st.stop()
        
        # Retrieve context
        with st.spinner("Retrieving context..."):
            context = retrieve_context(question, st.session_state.vectorstore)
        
        if not context:
            st.warning("No relevant context found. Using empty context.")
            context = "No relevant information found in the knowledge base."
        
        # Initialize state
        initial_state = {
            "question": question,
            "context": context,
            "generation": "",
            "reflection": "",
            "iteration": 0,
            "max_iterations": max_iterations,
            "reflection_history": []
        }
        
        # Build and run graph
        with st.spinner("Processing your question..."):
            graph = build_graph(llm)
            
            # Run the graph
            final_state = graph.invoke(initial_state)
        
        # Display results
        st.markdown("### Answer")
        st.write(final_state["generation"])
        
        # Show reflection details
        with st.expander("View Reflection Details"):
            st.write(f"**Iterations:** {final_state['iteration']}")
            st.write(f"**Final Reflection:** {final_state['reflection']}")
            
            if final_state.get("reflection_history"):
                st.write("**Reflection History:**")
                for i, reflection in enumerate(final_state["reflection_history"], 1):
                    st.write(f"Iteration {i}: {reflection}")
        
        # Show context used
        with st.expander("View Retrieved Context"):
            st.write(context)


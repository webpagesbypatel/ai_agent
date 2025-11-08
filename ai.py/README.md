# 🤖 RAG with Self-Correction Loop

A sophisticated Retrieval-Augmented Generation (RAG) system with an intelligent self-correction mechanism built using LangGraph, LangChain, HuggingFace, ChromaDB, and Streamlit. This application automatically evaluates and improves generated answers through iterative refinement.

## 📋 Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Usage Guide](#usage-guide)
- [How It Works](#how-it-works)
- [Adding to GitHub](#adding-to-github)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **Retrieval-Augmented Generation (RAG)**: Answers questions based on your custom knowledge base
- **Self-Correction Loop**: Automatically evaluates and improves answers through iterative refinement
- **Vector Database**: Uses ChromaDB for efficient semantic search and document retrieval
- **HuggingFace Integration**: Leverages pre-trained models for embeddings and text generation
- **Interactive UI**: Beautiful Streamlit interface for easy interaction
- **Reflection Mechanism**: Quality assurance through automatic answer evaluation
- **Configurable Iterations**: Control the maximum number of refinement iterations

## 🛠️ Technologies Used

- **LangChain**: Framework for building LLM applications
- **LangGraph**: State machine framework for building agent workflows
- **HuggingFace**: Pre-trained language models and embeddings
- **ChromaDB**: Vector database for document storage and retrieval
- **Streamlit**: Web framework for building interactive applications
- **Python 3.8+**: Programming language

## 📁 Project Structure

```
ai.py/
│
├── main.py                 # Main Streamlit application
├── agent.py                # LangGraph agent with prompts and state management
├── app.py                  # Planning prompt template
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
└── chroma_db/             # ChromaDB database directory (created automatically)
    └── ...
```

### File Descriptions

- **main.py**: Contains the Streamlit UI, graph construction, and main application logic
- **agent.py**: Defines the GraphState, prompts (RAG and reflection), and routing functions
- **app.py**: Contains the planning prompt for determining if retrieval is needed
- **requirements.txt**: Lists all Python package dependencies

## 📦 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8 or higher** (Python 3.10+ recommended)
- **pip** (Python package installer)
- **Git** (for version control, optional)

## 🔧 Installation

### Step 1: Clone or Download the Repository

If you're using Git:

```bash
git clone <your-repository-url>
cd ai.py
```

Or download and extract the project files to a directory.

### Step 2: Create a Virtual Environment (Recommended)

It's recommended to use a virtual environment to avoid conflicts with other projects:

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

Install all required packages using pip:

```bash
pip install -r requirements.txt
```

This will install:
- LangChain and LangGraph
- HuggingFace transformers and sentence-transformers
- ChromaDB
- Streamlit
- And all other dependencies

**Note**: The installation may take several minutes, especially when downloading PyTorch and model files.

## 🚀 Running the Application

### Start the Streamlit Server

Run the following command in your terminal:

```bash
streamlit run main.py
```

The application will start and automatically open in your default web browser at `http://localhost:8501`.

If the browser doesn't open automatically, navigate to:
```
http://localhost:8501
```

### Stop the Application

To stop the server, press `Ctrl+C` in the terminal where Streamlit is running.

## 📖 Usage Guide

### 1. Create a Knowledge Base

1. In the sidebar, find the "Knowledge Base" section
2. Paste or type your documents/text into the text area
3. Click the **"Create Knowledge Base"** button
4. Wait for the confirmation message

### 2. Initialize Models

1. In the sidebar, click the **"Initialize Models"** button
2. Wait for the models to load (this may take a few minutes on first run)
   - The embeddings model will download automatically
   - The language model will load into memory

### 3. Configure Settings

- Adjust the **"Max Iterations"** slider (1-5) to control how many times the system can refine an answer
- Default is 3 iterations

### 4. Ask Questions

1. Type your question in the chat input at the bottom of the main area
2. Press Enter or click send
3. The system will:
   - Retrieve relevant context from your knowledge base
   - Generate an initial answer
   - Evaluate the answer quality
   - Refine if needed (up to max iterations)
   - Display the final answer

### 5. View Details

- Click **"View Reflection Details"** to see:
  - Number of iterations used
  - Final reflection evaluation
  - Complete reflection history
- Click **"View Retrieved Context"** to see the source material used

## 🔄 How It Works

### Architecture Overview

1. **Question Input**: User asks a question
2. **Context Retrieval**: System searches the ChromaDB vector store for relevant information
3. **Answer Generation**: LLM generates an answer based on the retrieved context
4. **Reflection**: Another LLM evaluates the answer for:
   - Relevance to the question
   - Faithfulness to the context
5. **Decision**: System decides to:
   - **End**: If answer is complete and accurate
   - **Regenerate**: If answer needs improvement
6. **Iteration**: Process repeats up to the maximum iteration limit

### Self-Correction Loop

The self-correction mechanism uses LangGraph to create a state machine:

```
[Question] → [Retrieve Context] → [Generate Answer] → [Reflect] → [Decision]
                                                                    ↓
                                                          [End] or [Regenerate]
```

The reflection prompt checks:
- Does the answer directly respond to the question?
- Is the answer fully supported by the context?

If both conditions are met, the system outputs: "The answer is relevant and complete."

## 📤 Adding to GitHub

### Step 1: Create a GitHub Repository

1. Go to [GitHub](https://github.com) and sign in
2. Click the "+" icon in the top right, select "New repository"
3. Name your repository (e.g., "rag-self-correction")
4. Choose public or private
5. **Do NOT** initialize with README, .gitignore, or license (we'll add our own)
6. Click "Create repository"

### Step 2: Initialize Git in Your Project

Open your terminal in the project directory and run:

```bash
# Initialize Git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: RAG with Self-Correction Loop"
```

### Step 3: Create .gitignore File

Create a `.gitignore` file to exclude unnecessary files:

```bash
# Create .gitignore
echo "__pycache__/" > .gitignore
echo "*.pyc" >> .gitignore
echo "*.pyo" >> .gitignore
echo "*.pyd" >> .gitignore
echo ".Python" >> .gitignore
echo "venv/" >> .gitignore
echo "env/" >> .gitignore
echo ".env" >> .gitignore
echo "chroma_db/" >> .gitignore
echo "*.db" >> .gitignore
echo "*.sqlite3" >> .gitignore
echo ".streamlit/" >> .gitignore
echo ".DS_Store" >> .gitignore
echo "*.log" >> .gitignore
```

Or create it manually with these contents.

### Step 4: Connect to GitHub and Push

```bash
# Add remote repository (replace with your repository URL)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

### Step 5: Verify

1. Refresh your GitHub repository page
2. You should see all your files including the README.md
3. The README will automatically display on the repository homepage

### Optional: Add Repository Description

On GitHub, you can add:
- **Description**: "RAG system with self-correction loop using LangGraph, LangChain, and HuggingFace"
- **Topics**: `rag`, `langchain`, `langgraph`, `huggingface`, `chromadb`, `streamlit`, `ai`, `machine-learning`

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. ModuleNotFoundError: No module named 'langgraph'

**Solution:**
```bash
pip install langgraph
```

#### 2. ModuleNotFoundError: No module named 'langchain.prompts'

**Solution:** Update your imports to use `langchain_core.prompts`:
```python
from langchain_core.prompts import PromptTemplate
```

#### 3. Error loading embeddings: Could not import sentence_transformers

**Solution:**
```bash
pip install sentence-transformers
```

#### 4. Port Already in Use

If port 8501 is already in use:

**Solution:**
```bash
# Use a different port
streamlit run main.py --server.port 8502
```

#### 5. Out of Memory Errors

**Solution:**
- Use a smaller model in `main.py` (change `model_id`)
- Reduce the `max_iterations` setting
- Close other applications using memory

#### 6. Slow Model Loading

**Solution:**
- First-time model downloads can take time
- Models are cached after first download
- Consider using smaller models for faster loading

#### 7. ChromaDB Permission Errors

**Solution:**
- Ensure write permissions in the project directory
- Delete the `chroma_db` folder and recreate it

## 🔧 Configuration

### Changing the Language Model

Edit `main.py` and modify the `model_id` in the `initialize_llm()` function:

```python
model_id = "microsoft/DialoGPT-medium"  # Change this
```

Popular alternatives:
- `gpt2` (smaller, faster)
- `distilgpt2` (lightweight)
- `microsoft/DialoGPT-small` (smaller version)

### Changing the Embeddings Model

Edit `main.py` and modify the model in `initialize_embeddings()`:

```python
st.session_state.embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # Change this
)
```

## 📝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [LangChain](https://www.langchain.com/) for the LLM framework
- [LangGraph](https://github.com/langchain-ai/langgraph) for state machine workflows
- [HuggingFace](https://huggingface.co/) for pre-trained models
- [ChromaDB](https://www.trychroma.com/) for vector database
- [Streamlit](https://streamlit.io/) for the web framework

## 📧 Contact

For questions, issues, or suggestions, please open an issue on GitHub.

---

<img width="1919" height="968" alt="Screenshot 2025-11-08 165106" src="https://github.com/user-attachments/assets/b09e4c0f-d292-42b2-b949-78bbb1ff8b9b" />








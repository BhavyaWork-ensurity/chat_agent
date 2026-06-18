# Chat Agent with Long-Term Memory

A command-line AI chat agent that can remember important user information across conversations.

The project uses:

* LangChain
* Ollama
* ChromaDB
* Sentence Transformers

The agent extracts memory-worthy facts from user messages, stores them as vector embeddings, and retrieves relevant memories when generating future responses.

## Features

* Conversational AI powered by Ollama
* Long-term memory storage using ChromaDB
* Semantic memory retrieval using vector embeddings
* Automatic memory extraction from user messages
* Persistent memory across sessions
* Lightweight command-line interface

## Project Structure

```text
chat_agent/
│
├── main.py          # Application entry point
├── agent.py         # Chat and memory extraction logic
├── memory.py        # Memory storage and retrieval
├── config.py        # Configuration values
├── requirements.txt
└── chroma_db/       # Persistent memory database
```

## How It Works

### 1. User Sends a Message

```text
You: My favorite sport is football.
```

### 2. Memory Extraction

The agent determines whether the message contains information worth remembering.

Example extracted memory:

```json
{
  "remember": true,
  "memory": "User's favorite sport is football"
}
```

### 3. Memory Storage

The memory is converted into an embedding using Sentence Transformers and stored in ChromaDB.

### 4. Memory Retrieval

When the user asks a related question later, relevant memories are retrieved through semantic search.

Example:

```text
You: What sport do I like?
```

Retrieved memory:

```text
User's favorite sport is football
```

### 5. Response Generation

The retrieved memories are included in the prompt before sending it to the language model.

## Installation

### Clone the Repository

```bash
git clone https://github.com/BhavyaWork-ensurity/chat_agent.git
cd chat_agent
```

### Create a Virtual Environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Install Ollama

Download and install Ollama:

https://ollama.com

Pull the model configured in `config.py`.

Example:

```bash
ollama pull llama3
```

## Configuration

Edit `config.py`:

```python
MODEL_NAME = "llama3"
CHROMA_PATH = "./chroma_db"
```

## Running the Project

```bash
python main.py
```

Example:

```text
Agent ready

You: I live in Mumbai.

Assistant: Nice to meet you.

You: Where do I live?

Assistant: You previously mentioned that you live in Mumbai.
```

## Technologies Used

* Python
* LangChain
* Ollama
* ChromaDB
* Sentence Transformers

## Future Improvements

* Web interface using FastAPI
* Telegram or Discord integration
* User authentication
* Memory editing and deletion
* Better memory extraction prompts
* Multi-user support
* Hybrid search and retrieval

## Learning Goals

This project was built to explore:

* Retrieval-Augmented Generation (RAG)
* Long-term memory systems
* Vector databases
* Semantic search
* Local LLM deployment with Ollama
* AI agent architectures

## License

This project is open source and available under the MIT License.

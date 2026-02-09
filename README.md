# rag-interview-qa-system
RAG-based Q&amp;A system using LangChain, ChromaDB, HuggingFace embeddings, and Gemini Flash to answer interview-style questions from custom text data

# RAG-based Interview Q&A System

This project implements a Retrieval-Augmented Generation (RAG) pipeline that answers interview-style questions using a custom text-based knowledge base.

The system retrieves semantically relevant content from a vector database (ChromaDB) and uses a Large Language Model (Google Gemini Flash) to generate concise, human-readable answers.


## Tech Stack

- Python
- LangChain (community + classic)
- ChromaDB (vector database)
- HuggingFace Sentence Transformers (embeddings)
- Google Gemini 2.5 Flash (LLM)



## Dataset

- Text-based interview question–answer files
- Topics include:
  - Python
  - SQL
  - Machine Learning
  - Generative AI
- Files are loaded using LangChain’s `DirectoryLoader` and `TextLoader`



## System Architecture

1. Load text documents from local directory
2. Split documents into chunks
3. Generate embeddings using HuggingFace models
4. Store embeddings in ChromaDB
5. Retrieve top-k relevant chunks for a query
6. Use Gemini LLM to synthesize a final answer



## Chunking Strategy

Initial experiments with small chunk sizes caused fragmented and incomplete answers.

Final configuration:

-> chunk_size = 1000
-> chunk_overlap = 100

This preserves full question–answer context and improves answer coherence for interview-style data.



## Why Use an LLM?

While the vector database retrieves relevant text chunks, it returns raw content.

The LLM is required to:
- Summarize retrieved chunks
- Merge multiple sources
- Produce human-readable answers
- Handle slight variations in user queries

Without the LLM, the system only performs semantic search, not question answering.



## Limitations

- Dataset contains mostly basic interview questions
- No formal retrieval evaluation metrics implemented
- Character-based chunking may not scale for long-form documents
- Answers are limited to the provided dataset



## Future Improvements

- Chunk data by question–answer structure instead of characters
- Add deeper explanations and code snippets to the dataset
- Implement retrieval evaluation (Precision@K)
- Migrate to LCEL-style LangChain chains



## Disclaimer

This project is built for learning and experimentation purposes.

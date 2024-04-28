# LangChain

LangChain is an open-source framework designed to assist developers in creating applications using large language models (LLMs). It offers Python- and JavaScript-based libraries, providing tools and APIs to streamline the development of LLM-driven applications such as chatbots and virtual agents.

## Overview

LangChain facilitates the connection of external data and computation to LLMs, enabling the creation of dynamic and responsive applications. Below is a quick overview of LangChain's key features:

### Overview
**Building with LangChain:**
- LangChain connects external data to LLMs for applications.
  1. Simple LLM chain.
  2. Retrieval chain.
  3. Conversation retrieval chain.
  4. Agent utilizing LLM dynamically.

1. **Simple LLM chain**: Generates responses based on information within the prompt template.
    - Uses models via API (e.g., OpenAI) or local open-source models (e.g., Ollama).
    - Requires initialization and API key setup.
    - Prompt templates guide responses.
   
3. **Retrieval chain**: Retrieves data from an external database and integrates it into the prompt template, enhancing response capabilities.
  - Purpose: Provides additional context to LLM for answering a question.
  - Use: When too much data for LLM, retriever fetches relevant pieces.
  - Process: Retrieve relevant documents, pass them to LLM.
  - Retriever Source: Can be from SQL table, internet, etc.
  - Example: Populating vector store as retriever.
  - Resources: Refer to documentation for vector stores.
  - Fetches relevant documents for LLM context.
  - Utilizes retriever to fetch and pass data.
  - Indexes data using vector store for retrieval.
4. **Conversation retrieval chain**: Utilizes chat history to create conversational interactions, enabling the model to remember and respond contextually.
  - Extends retrieval to handle chatbot-like interactions.
  - Dynamically retrieves documents based on conversation history.
5. **Agent**: Employs an LLM to determine whether fetching additional data is necessary to answer questions, optimizing response accuracy and relevance.
  - Dynamic decision-making by LLM.
  - Tools provided access to retriever and search.
  - LangChain Hub and OpenAI integration for predefined prompts.

**LangSmith:**
- Multiple steps and LLM calls in LangChain applications necessitate inspection.
- LangSmith aids in monitoring complex chains or agents.
- Optional but beneficial for understanding chain dynamics.

**Serving with LangServe:**
- LangServe facilitates deploying LangChain apps as REST APIs.
- Optional but simplifies deployment process.

### Step-by-Step Process:

1. **Setting up LangSmith:**
   - Sign up and set environment variables for logging traces.

2. **Building with LangChain:**
   - Create applications connecting data sources to LLMs.
   - Implement quickstart steps: LLM chain, retrieval chain, conversation retrieval chain, and agent.

3. **LLM Chain:**
   - Initialize LLM models (e.g., OpenAI) and prompt templates.
   - Combine prompt and LLM for response generation.

4. **Retrieval Chain:**
   - Load and index data for retrieval using a retriever.
   - Create a chain to fetch and pass relevant documents to LLM.

5. **Conversation Retrieval Chain:**
   - Modify retrieval to consider conversation history.
   - Extend LLM chain for dynamic interaction.

6. **Agent:**
   - Configure tools for retriever and search.
   - Utilize LangChain Hub and OpenAI integration for agent setup.

7. **Serving with LangServe:**
   - Deploy LangChain applications as REST APIs using LangServe.
   - Simplify deployment process for LangChain apps.

## Open source Models:

1. **OLLAMA**: Developed by Meta for open language learning research.
2. **Cohere**: Offers models for various natural language tasks.
3. **Hugging Face**: Provides diverse pre-trained models for NLP tasks.
4. **Google Gemini**: Google's language model for natural language understanding.
5. **Mixtral**: Supports tasks like translation and summarization.
6. **GPT-NeoX-20B**: Variant of GPT with 20 billion parameters.
7. **Bloom**: Focuses on efficiency and scalability.
8. **Falcon**: Open-source model for text generation.

## Selection Criteria:

Consider task requirements, model performance, available resources, and compatibility when selecting the best model.




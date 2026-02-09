# How to Build a Retrieval-Augmented Generation (RAG) Chatbot
 
### Introduction:
This document provides step-by-step instructions to build a RAG chatbot and system to allow users to chat with their own pdf files. This guide will cover environment setup and chatbot database initialization. This guide will not cover advanced Machine Learning theory, or custom model training.
 
### Audience:
This document is meant for anyone that wants to leverage Large Language Models (LLMs) to analyze and interact with their personal files, without uploading them to public, third-party interfaces that have training data on information that is outside of what has been provided by the user. No advanced coding degree is required, however users should have a basic comfort level with installing software via a terminal. In addition, users to be sure to note that while this system provides a high level of accuracy for their uploaded files, as it does not have access to the data that public third-party chatbots have, this tool is designed for context-specific retrieval, and not general-purpose knowledge.
 
### How RAG Works:
LLMs are trained on vast amounts of public data, but they don’t know exactly what is in your files and folders. RAG bridges this gap, retrieving information specifically from your files without you having to retrain an AI, something that is both expensive and difficult, and instead acts as a search engine of your files for the AI.

**Why Use RAG:**
- Privacy (your data never leaves your computer)
- Accuracy (your answers are based exactly on the information you provide)
- Up-to-Date (as soon as you add more files, your AI knows and can use that)
 
### Prerequisites and Environment Setup:
This project requires specific hardware and software configurations to work. This set of instructions will ensure that the RAG runs smoothly on a 16GB+ MacBook, however the RAG chatbot can be run on Linux, Windows, or other operating systems as well and this is simply the model the guide will demonstrate with. 

**Software Installation Guide:**
1. To download Ollama, go to [Ollama.com](https://ollama.com) and click the download button. Ollama is the open-source tool that allows users to download, run, and manage LLMs locally on Windows, MacOS, and Linux while ensuring data privacy.
2. Once downloaded, double-click on the downloaded file and then drag the Ollama app icon to the Applications folder.
3. Open the Terminal on your computer (Cmd-Space to Spotlight Search) and run these two commands:
   ```bash
   ollama pull llama3.1:8b
   ollama pull nomic-embed-text
   ```
   The first command gets the “brain” of your AI, while the second acts as the translator between you and the AI.

   > **Note for advanced users:** Different models offer various benefits regarding speed and reasoning. You can research and choose different models at [Ollama's library](https://ollama.com/library) based on your hardware capabilities.

4. Ensure that Python 3.10 or higher is installed on your computer by typing `python3 --version` in your Terminal. If it is not installed, follow the steps below:
   * a. Go to [Python.org](https://python.org) and download the macOS 64-bit universal2 installer for the latest stable version (currently Python 3.14.3).
   * b. Open the downloaded .pkg file and follow the on-screen prompts. You will need to enter your Mac password.
   * c. Once the installation finishes, a folder will open in Finder. Double-click the file named Install Certificates.command. This allows Python to download the AI models securely.
   * d. Open your Terminal and type `python3 --version`. You should see your Python version in the output.

5. If you do not have Visual Studio Code installed, follow the steps below:
   * a. Open your browser and go to [code.visualstudio.com](https://code.visualstudio.com)
   * b. Click the downward arrow next to the "Download for Mac" button.
   * c. Select the Apple Silicon (zip) or Universal version. Apple M chip users should select the Apple Silicon build  for fastest performance.
   * d. Locate the downloaded .zip file in your Downloads folder and double-click it. This will extract the "Visual Studio Code" application icon.
   * e. Drag the Visual Studio Code icon into your Applications folder, the same way you did for Ollama.
   * f. Open VS Code from your Applications folder. When macOS asks if you want to open an app downloaded from the internet, click Open.
   * g. Press `Cmd + Shift + P` to open the Command Palette.
   * h. Type "shell command".
   * i. Select Shell Command: Install 'code' command in PATH. You will see a small notification in the bottom right corner confirming the command was successfully installed.
   
6. To create your workspace, create a folder on your Desktop named `rag-project`.
7. Open this folder in VS Code, and create a virtual environment by opening the Terminal in VS Code (Click New Terminal, found in the top right bar of VS Code) and typing:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
8. In order to install the necessary libraries, run:
   ```bash
   pip install langchain-ollama langchain-chroma langchain-community pypdf
   ```
   **What do each of these do:**
   - **langchain-ollama:** Integration for Ollama LLMs.
   - **langchain-chroma:** Handles the vector database storage.
   - **langchain-community:** Third-party integrations for data loading.
   - **pypdf:** Specifically used to read and parse PDF files.
 
### Data Preparation:
Before the AI can read your documents, you must organize them so the code can find them.
1. Inside your `rag-project` folder in VS Code, create a new folder named `data`.
2. Ensure the PDF files you want to “talk” to are not password-protected. Copy the PDFs into this `data` folder. 
3. Also inside your `rag-project` folder in VS Code, create a new file named `ingest.py`.
4. Copy this code block into the `ingest.py` file:
```python
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
 
# Point to the folder with all the PDFs
loader = PyPDFDirectoryLoader("data")
docs = loader.load()
 
print(f"Loaded {len(docs)} pages from your folder.")
 
# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)
```
 
### Vector Database Initialization:
The Vector Database acts as the "Librarian" for your AI. It converts text into mathematical coordinates (vectors). 
1. Copy this code to the bottom of `ingest.py`, after the text splitting:
```python
# Create the numeric map (Embeddings) using Nomic
embeddings = OllamaEmbeddings(model="nomic-embed-text")
 
# Save everything to your local database
vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory="./chroma_db"
)
 
print("All PDFs are now successfully indexed in your local database!")
```
2. In your VS Code Terminal or Mac Terminal, run `python3 ingest.py`. You may see messages like "Ignoring wrong pointing object." You can ignore these; the script is bypassing minor PDF formatting errors.
3. Once the Terminal returns to the command prompt, verify that a new folder named `chroma_db` has appeared in your `rag-project` folder, either in your sidebar in VS Code or in your Documents folder in Finder.
 
### Building the Query Loop:
The query loop is the interface that allows you to have a back-and-forth conversation with your files.
1. Inside your `rag-project` folder in VS Code, create a new file named `chat.py`.
2. Copy this code in `chat.py`:
```python
import logging
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
 
# Mute warnings from PDF loading for a cleaner chat experience
logging.getLogger("pypdf").setLevel(logging.ERROR)
 
# Setup your local "Brain" and "Librarian"
# nomic handles the mathematical map of your PDFs
embeddings = OllamaEmbeddings(model="nomic-embed-text")
 
# Connect to your existing database created by ingest.py
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
 
# llama3.1:8b is the high-performance model for 16GB Macs
llm = ChatOllama(model="llama3.1:8b", temperature=0)
 
# Define the "Rules" for the AI (The Prompt Template)
# This forces the AI to use ONLY your PDFs and admit when it's lost
template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know. 
Do not try to make up an answer.
 
Context: {context}
Question: {question}
Helpful Answer:"""
 
QA_PROMPT = PromptTemplate(
    template=template, 
    input_variables=["context", "question"]
)
 
# Build the RAG Chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(search_kwargs={"k": 3}), # Pulls 3 most relevant chunks
    return_source_documents=True, # Allows us to see which PDFs were read
    chain_type_kwargs={"prompt": QA_PROMPT}
)
 
# The Interactive Chat Loop
print("\n--- My Very Own RAG Bot ---")
print("(Type 'quit' to exit)")
 
while True:
    user_query = input("\nQuestion: ")
    if user_query.lower() in ['quit', 'exit']:
        break
 
    print("Searching your files...")
    response = rag_chain.invoke({"query": user_query})
    
    # Display the result
    print(f"\nAnswer: {response['result']}")
 
    # Display the Sources
    print("\nSources used:")
    sources = set() # Use a set to avoid duplicate file names
    for doc in response["source_documents"]:
        source_name = doc.metadata.get('source', 'Unknown')
        page_num = doc.metadata.get('page', 'N/A')
        sources.add(f"- {source_name} (Page {page_num})")
    
    for s in sources:
        print(s)
```
 
### Testing and Validation:
Testing ensures that your RAG system is actually reading your files and not just using its general knowledge.
1. In your VS Code Terminal or Mac Terminal, run `python3 chat.py`.
2. Ask a question that you know is answered in your PDF to verify that the answer matches your document and the correct page number is cited.
3. Ask a question that is not in your PDF. The AI should respond with "I don't know" or a similar refusal based on your prompt rules.
4. If you want to, you can open the Activity Monitor on your Mac and look at the "Memory" tab to see how Ollama manages the 8B model within your **16GB of RAM.**

---

### Conclusion:
By following this guide, you have successfully built a private, local RAG chatbot. This system allows you to gain insights from your personal data with high accuracy and full privacy, bypassing the need for third-party cloud interfaces. However, you can fine-tune the model by adjusting the temperature value in chat.py; increasing it encourages creativity and the use of outside information not simply in your uploaded pdfs but raises the risk of hallucinations (inaccurate information), while decreasing it makes the answers more precise and factual.

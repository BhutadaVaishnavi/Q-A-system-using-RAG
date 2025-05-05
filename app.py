
## Pdf reader
from langchain_community.document_loaders import PyPDFLoader
loader=PyPDFLoader('Indian_Hotels_data.pdf')
docs=loader.load()

from langchain_community.llms import Ollama
llm = Ollama(model="llama3") 

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
documents=text_splitter.split_documents(docs)
documents[:5]

## Vector Embedding And Vector Store
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Use a suitable embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma.from_documents(documents, embeddings)


# query = "Show me best hotels in goa"
# retireved_results=db.similarity_search(query)
# print(retireved_results[0].page_content)
from langchain.chains.question_answering import load_qa_chain
chain = load_qa_chain(llm, chain_type="stuff")

import streamlit as st
st.title("📄 PDF Chatbot using LLaMA3 + HuggingFace + Chroma")

query = st.text_input("Ask a question about the PDF:")

if query:
    # Search and answer
    retrieved_docs = db.similarity_search(query)
    answer = chain.run(input_documents=retrieved_docs, question=query)

    # Display answer
    st.markdown("### 🔍 Answer:")
    st.write(answer)
    
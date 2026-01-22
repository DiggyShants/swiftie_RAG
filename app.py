onimport streamlit as st
import os
from dotenv import load_dotenv, find_dotenv

# Modern Modular Imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

# 1. Environment Setup
load_dotenv(find_dotenv())

st.set_page_config(page_title="Swiftie AI", page_icon="🎤")
st.title("Swiftie AI: Question the Data 💃")

# 2. Connection to Vector Store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = PineconeVectorStore(
    index_name="taylor-swift-rag", 
    embedding=embeddings
)

# 3. Define the Prompt (The AI's Personality)
# This template tells the AI how to use the retrieved lyrics
system_prompt = (
    "You are an expert Swiftie. Use the provided Taylor Swift lyrics to answer questions. "
    "If you don't know the answer based on the lyrics, say so using a Taylor Swift pun. "
    "Keep answers concise and always cite the song and album name."
    "\n\n"
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# 4. Build the Modern Chain
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)

# This chain handles combining the retrieved documents into the prompt
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# This is the main chain that connects the retriever to the QA logic
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 5. The User Interface
user_input = st.text_input("Ask about the lyrics:", placeholder="What does she say about New York?")

if user_input:
    with st.spinner("Searching the vault..."):
        # The new chain uses 'input' as the key and returns a dict with 'answer'
        response = rag_chain.invoke({"input": user_input})
        
        st.write("### The Verdict")
        st.info(response["answer"]) # Note: the key is now "answer", not "result"
        
        # Show the Sources
        st.write("---")
        st.write("**Sources used:**")
        for doc in response["context"]:
            st.caption(f"- {doc.metadata['track_name']} ({doc.metadata['album_name']})")



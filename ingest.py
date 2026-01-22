import pandas as pd
import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# 1. THE MAC FIX: Force-load the .env file
# find_dotenv() searches the folder structure specifically for that file
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

# 2. DIAGNOSTIC CHECK
PINECONE_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

print("--- System Check ---")
print(f"Directory: {os.getcwd()}")
print(f".env found at: {dotenv_path}")
print(f"Pinecone Key Detected: {'✅ YES' if PINECONE_KEY else '❌ NO'}")
print(f"OpenAI Key Detected:   {'✅ YES' if OPENAI_KEY else '❌ NO'}")
print("--------------------")

def run_ingestion():
    # 1. Load the Taylor Swift lyrics
    print("Reading lyrics from CSV...")
    df = pd.read_csv("taylor_lyrics.csv")

    # 2. Chunking: We group every 4 lines of a song together.
    # This gives the AI enough context (a full verse) to understand the meaning.
    print("Grouping lyrics into chunks...")
    df['chunk_id'] = df.index // 4 
    chunks = df.groupby(['track_name', 'album_name', 'chunk_id'])['lyric'].apply(' '.join).reset_index()

    texts = chunks['lyric'].tolist()
    # Metadata helps the AI tell you which song/album the answer came from
    metadatas = chunks[['track_name', 'album_name']].to_dict('records')

    # 3. Connect to Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "taylor-swift-rag"

    # Create the index if it doesn't exist
    if index_name not in pc.list_indexes().names():
        print(f"Creating index: {index_name}...")
        pc.create_index(
            name=index_name,
            dimension=1536, # Standard size for OpenAI embeddings
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )

    # 4. Embed and Upload
    # This step converts text into numbers (vectors) and sends them to Pinecone
    print("Converting lyrics to vectors and uploading to Pinecone (this may take a minute)...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vector_store = PineconeVectorStore.from_texts(
        texts=texts,
        embedding=embeddings,
        index_name=index_name,
        metadatas=metadatas
    )

    print("Success! Taylor's lyrics are now stored in your vector database.")

if __name__ == "__main__":
    run_ingestion()
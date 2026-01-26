import pandas as pd
import os
import time
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# PDF processing
import pdfplumber

# =============================================================================
# 1. ENVIRONMENT SETUP
# =============================================================================
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

PINECONE_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

print("--- System Check ---")
print(f"Directory: {os.getcwd()}")
print(f".env found at: {dotenv_path}")
print(f"Pinecone Key Detected: {'✅ YES' if PINECONE_KEY else '❌ NO'}")
print(f"OpenAI Key Detected:   {'✅ YES' if OPENAI_KEY else '❌ NO'}")
print("--------------------")

INDEX_NAME = "swiftierag"

# =============================================================================
# 2. LYRICS INGESTION
# =============================================================================
def process_lyrics(csv_path: str) -> tuple[list[str], list[dict]]:
    """
    Process Taylor Swift lyrics CSV into chunks with metadata.
    Returns (texts, metadatas) with source="lyrics"
    """
    print("\n📝 Processing lyrics...")
    df = pd.read_csv(csv_path)
    
    # Group every 4 lines together (roughly a verse)
    df['chunk_id'] = df.index // 4
    chunks = df.groupby(['track_name', 'album_name', 'chunk_id'])['lyric'].apply(' '.join).reset_index()
    
    texts = chunks['lyric'].tolist()
    
    # Build metadata with source="lyrics"
    metadatas = []
    for _, row in chunks.iterrows():
        metadatas.append({
            'track_name': row['track_name'],
            'album_name': row['album_name'],
            'source': 'lyrics'  # <-- NEW: Source label
        })
    
    print(f"   ✅ Created {len(texts)} lyric chunks")
    return texts, metadatas

# =============================================================================
# 3. WIKIPEDIA PDF INGESTION
# =============================================================================
def process_wikipedia_pdf(pdf_path: str, chunk_size: int = 500) -> tuple[list[str], list[dict]]:
    """
    Process Wikipedia PDF into chunks with metadata.
    Returns (texts, metadatas) with source="wikipedia"
    
    Args:
        pdf_path: Path to the Wikipedia PDF
        chunk_size: Approximate number of characters per chunk
    """
    print(f"\n📄 Processing Wikipedia PDF: {pdf_path}")
    
    # Extract all text from PDF
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        print(f"   Found {len(pdf.pages)} pages")
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n\n"
    
    print(f"   Extracted {len(full_text)} characters of text")
    
    # Split into chunks by sentences/paragraphs
    # Try multiple splitting strategies
    import re
    
    # Split on double newlines, single newlines, or sentence endings
    segments = re.split(r'\n\n+|\n(?=[A-Z])|(?<=[.!?])\s+(?=[A-Z])', full_text)
    segments = [s.strip() for s in segments if s.strip() and len(s.strip()) > 20]
    
    print(f"   Found {len(segments)} text segments")
    
    texts = []
    metadatas = []
    current_chunk = ""
    
    for segment in segments:
        # If adding this segment exceeds chunk size, save current chunk
        if len(current_chunk) + len(segment) > chunk_size and current_chunk:
            texts.append(current_chunk.strip())
            metadatas.append({
                'source': 'wikipedia',
                'content_type': 'biography'
            })
            current_chunk = segment
        else:
            current_chunk += " " + segment if current_chunk else segment
    
    # Don't forget the last chunk
    if current_chunk.strip():
        texts.append(current_chunk.strip())
        metadatas.append({
            'source': 'wikipedia',
            'content_type': 'biography'
        })
    
    print(f"   ✅ Created {len(texts)} Wikipedia chunks")
    return texts, metadatas

# =============================================================================
# 4. PINECONE INDEX MANAGEMENT
# =============================================================================
def delete_index_if_exists(pc: Pinecone, index_name: str):
    """Delete the index if it exists (for clean re-ingestion)."""
    if index_name in pc.list_indexes().names():
        print(f"\n🗑️  Deleting existing index: {index_name}")
        pc.delete_index(index_name)
        # Wait for deletion to complete
        time.sleep(5)
        print("   ✅ Index deleted")

def create_index(pc: Pinecone, index_name: str):
    """Create a new Pinecone index."""
    if index_name not in pc.list_indexes().names():
        print(f"\n🔨 Creating index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=1536,  # OpenAI text-embedding-3-small dimension
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        # Wait for index to be ready
        time.sleep(10)
        print("   ✅ Index created")

# =============================================================================
# 5. MAIN INGESTION FUNCTION
# =============================================================================
def run_ingestion(
    lyrics_csv: str = "taylor_lyrics.csv",
    wikipedia_pdf: str = None,
    delete_existing: bool = True
):
    """
    Main ingestion function that processes lyrics and optionally Wikipedia PDF.
    
    Args:
        lyrics_csv: Path to the Taylor Swift lyrics CSV
        wikipedia_pdf: Path to Wikipedia PDF (optional)
        delete_existing: Whether to delete existing index first (recommended)
    """
    # Connect to Pinecone
    pc = Pinecone(api_key=PINECONE_KEY)
    
    # Optionally delete existing index for clean start
    if delete_existing:
        delete_index_if_exists(pc, INDEX_NAME)
    
    # Create index
    create_index(pc, INDEX_NAME)
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Collect all texts and metadata
    all_texts = []
    all_metadatas = []
    
    # 1. Process lyrics
    if os.path.exists(lyrics_csv):
        lyrics_texts, lyrics_meta = process_lyrics(lyrics_csv)
        all_texts.extend(lyrics_texts)
        all_metadatas.extend(lyrics_meta)
    else:
        print(f"⚠️  Lyrics file not found: {lyrics_csv}")
    
    # 2. Process Wikipedia PDF if provided
    if wikipedia_pdf and os.path.exists(wikipedia_pdf):
        wiki_texts, wiki_meta = process_wikipedia_pdf(wikipedia_pdf)
        all_texts.extend(wiki_texts)
        all_metadatas.extend(wiki_meta)
    elif wikipedia_pdf:
        print(f"⚠️  Wikipedia PDF not found: {wikipedia_pdf}")
    
    # 3. Upload to Pinecone
    if all_texts:
        print(f"\n🚀 Uploading {len(all_texts)} total chunks to Pinecone...")
        
        vector_store = PineconeVectorStore.from_texts(
            texts=all_texts,
            embedding=embeddings,
            index_name=INDEX_NAME,
            metadatas=all_metadatas
        )
        
        print("\n✨ SUCCESS! All content has been ingested.")
        print(f"   Total chunks: {len(all_texts)}")
        print(f"   - Lyrics chunks: {len([m for m in all_metadatas if m.get('source') == 'lyrics'])}")
        print(f"   - Wikipedia chunks: {len([m for m in all_metadatas if m.get('source') == 'wikipedia'])}")
    else:
        print("❌ No content to ingest!")

# =============================================================================
# 6. ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    # Update these paths as needed
    LYRICS_CSV = "taylor_lyrics.csv"
    WIKIPEDIA_PDF = "Taylor_Swift.pdf"
    
    run_ingestion(
        lyrics_csv=LYRICS_CSV,
        wikipedia_pdf=WIKIPEDIA_PDF,
        delete_existing=True  # Recreates index with correct 1536 dimensions
    )

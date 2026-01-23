import streamlit as st
import os
import random
from datetime import datetime
from dotenv import load_dotenv, find_dotenv

# Modern Modular Imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

# =============================================================================
# 1. ENVIRONMENT & PAGE CONFIGURATION
# =============================================================================
load_dotenv(find_dotenv())

# Era-based color themes
ERA_THEMES = {
    "All": {"primary": "#FF6B6B", "background": "#1a1a2e", "accent": "#FFD93D"},
    "Taylor Swift": {"primary": "#7CB342", "background": "#1B5E20", "accent": "#C5E1A5"},
    "Fearless": {"primary": "#FFD700", "background": "#3d3d00", "accent": "#FFF59D"},
    "Speak Now": {"primary": "#9C27B0", "background": "#2a1a30", "accent": "#E1BEE7"},
    "Red": {"primary": "#D32F2F", "background": "#2a1a1a", "accent": "#FFCDD2"},
    "1989": {"primary": "#03A9F4", "background": "#1a2a3d", "accent": "#B3E5FC"},
    "reputation": {"primary": "#212121", "background": "#0a0a0a", "accent": "#757575"},
    "Lover": {"primary": "#FF69B4", "background": "#2d1a2a", "accent": "#FFB6C1"},
    "folklore": {"primary": "#9E9E9E", "background": "#1a1a1a", "accent": "#E0E0E0"},
    "evermore": {"primary": "#8D6E63", "background": "#1a1612", "accent": "#D7CCC8"},
    "Midnights": {"primary": "#1A237E", "background": "#0d0d1a", "accent": "#7986CB"},
    "TTPD": {"primary": "#F5F5DC", "background": "#1a1a18", "accent": "#FFFAF0"},
}

# Fun facts and trivia for "Surprise Me" feature
TAYLOR_TRIVIA = [
    "Taylor wrote 'Love Story' in just 20 minutes on her bedroom floor!",
    "The 10-minute version of 'All Too Well' was the original draft.",
    "Taylor hides Easter eggs in almost everything she releases.",
    "'Betty' is from the perspective of James—a male character.",
    "The 'Fearless' re-recording broke multiple records on release day.",
    "'exile' was recorded with Taylor and Bon Iver in separate locations.",
    "Taylor's lucky number is 13—look for it everywhere!",
    "'cardigan', 'august', and 'betty' tell the same story from different perspectives.",
]

def get_era_css(era: str) -> str:
    """Generate CSS based on selected era theme."""
    theme = ERA_THEMES.get(era, ERA_THEMES["All"])
    return f"""
    <style>
        .stApp {{
            background-color: {theme['background']};
        }}
        .stButton > button {{
            background-color: {theme['primary']};
            color: white;
            border: none;
            border-radius: 20px;
            padding: 0.5rem 1rem;
            transition: all 0.3s ease;
        }}
        .stButton > button:hover {{
            background-color: {theme['accent']};
            color: black;
            transform: scale(1.05);
        }}
        .chat-message {{
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }}
        .user-message {{
            background-color: {theme['primary']}33;
            border-left: 4px solid {theme['primary']};
        }}
        .assistant-message {{
            background-color: {theme['accent']}22;
            border-left: 4px solid {theme['accent']};
        }}
        h1 {{
            color: {theme['primary']} !important;
        }}
        .source-expander {{
            background-color: {theme['primary']}11;
            border-radius: 8px;
            padding: 0.5rem;
            margin: 0.25rem 0;
        }}
        .sidebar .stSelectbox label {{
            color: {theme['accent']};
        }}
        .theme-button {{
            margin: 0.25rem;
        }}
    </style>
    """

# =============================================================================
# 2. SESSION STATE INITIALIZATION
# =============================================================================
def initialize_session_state():
    """Initialize all session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_era" not in st.session_state:
        st.session_state.selected_era = "All"
    if "similarity_threshold" not in st.session_state:
        st.session_state.similarity_threshold = 0.7
    if "num_sources" not in st.session_state:
        st.session_state.num_sources = 5

initialize_session_state()

# =============================================================================
# 3. PAGE SETUP & STYLING
# =============================================================================
st.set_page_config(
    page_title="Swiftie AI",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply era-based styling
st.markdown(get_era_css(st.session_state.selected_era), unsafe_allow_html=True)

# =============================================================================
# 4. SIDEBAR CONFIGURATION
# =============================================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/f/f8/Taylor_Swift_-_1989_%28Taylor%27s_Version%29.png", width=150)
    st.title("🎸 Swiftie Settings")
    
    st.markdown("---")
    
    # Era/Album Filter
    st.subheader("🎵 Filter by Era")
    albums = list(ERA_THEMES.keys())
    selected_album = st.selectbox(
        "Select an album:",
        albums,
        index=albums.index(st.session_state.selected_era),
        help="Filter lyrics by specific album or search all"
    )
    st.session_state.selected_era = selected_album
    
    st.markdown("---")
    
    # Retrieval Settings
    st.subheader("⚙️ Search Settings")
    
    st.session_state.num_sources = st.slider(
        "Number of sources to retrieve:",
        min_value=1,
        max_value=10,
        value=st.session_state.num_sources,
        help="How many lyric passages to search through"
    )
    
    st.session_state.similarity_threshold = st.slider(
        "Similarity threshold:",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.similarity_threshold,
        step=0.05,
        help="Higher = more strict matching, Lower = broader results"
    )
    
    use_mmr = st.checkbox(
        "Use diverse results (MMR)",
        value=True,
        help="Maximum Marginal Relevance reduces redundant results"
    )
    
    st.markdown("---")
    
    # Conversation Controls
    st.subheader("💬 Conversation")
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("📥 Export Conversation", use_container_width=True):
        if st.session_state.messages:
            export_text = f"# Swiftie AI Conversation\n"
            export_text += f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            for msg in st.session_state.messages:
                role = "You" if msg["role"] == "user" else "Swiftie AI"
                export_text += f"**{role}:** {msg['content']}\n\n"
            st.download_button(
                label="💾 Download as Markdown",
                data=export_text,
                file_name=f"swiftie_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
                use_container_width=True
            )
        else:
            st.info("No messages to export yet!")
    
    st.markdown("---")
    
    # Fun Stats
    st.subheader("📊 Session Stats")
    st.metric("Questions Asked", len([m for m in st.session_state.messages if m["role"] == "user"]))
    st.metric("Current Era", st.session_state.selected_era)

# =============================================================================
# 5. VECTOR STORE & RETRIEVER SETUP
# =============================================================================
@st.cache_resource
def get_vector_store():
    """Initialize and cache the vector store connection."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return PineconeVectorStore(
        index_name="taylor-swift-rag",
        embedding=embeddings
    )

vector_store = get_vector_store()

def get_retriever(album_filter: str, k: int, use_mmr: bool, score_threshold: float):
    """Create a retriever with the specified settings."""
    search_kwargs = {"k": k}
    
    # Add album filter if not "All"
    if album_filter != "All":
        search_kwargs["filter"] = {"album_name": album_filter}
    
    if use_mmr:
        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={**search_kwargs, "fetch_k": k * 3}
        )
    else:
        return vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={**search_kwargs, "score_threshold": score_threshold}
        )

# =============================================================================
# 6. SYSTEM PROMPT & CHAIN SETUP
# =============================================================================
SYSTEM_PROMPT = """You are a passionate Taylor Swift expert and lyric analyst. Your role is to help fans explore Taylor's discography by answering questions using the retrieved lyrics provided below.

## How to respond:
- Ground your answers in the specific lyrics provided in the context
- Quote relevant lyrics directly using quotation marks
- Always cite the song title and album in parentheses, e.g., (All Too Well, Red)
- Connect themes across songs when relevant
- Match Taylor's storytelling energy—be vivid and emotionally resonant
- If discussing songwriting craft, analyze rhyme schemes, metaphors, and narrative techniques

## Handling limitations:
- If the retrieved lyrics don't contain relevant information, acknowledge this honestly with a playful Taylor reference (e.g., "I knew you were trouble when you asked that—these lyrics don't quite cover it!")
- Never invent or hallucinate lyrics that aren't in the provided context
- If a question is ambiguous, ask for clarification about which era or theme they mean
- If asked about something outside Taylor Swift's music, gently redirect to relevant lyrics

## Tone:
- Enthusiastic but not over-the-top
- Analytical when discussing songwriting craft
- Warm and welcoming to all fans, whether casual listeners or vault-key holders
- Use occasional Taylor-isms and references naturally (but don't force them)

## Formatting:
- Use markdown for emphasis when quoting lyrics
- Keep responses focused and avoid unnecessary padding
- If multiple songs are relevant, organize your response clearly

Context:
{context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
])

@st.cache_resource
def get_llm():
    """Initialize and cache the LLM."""
    return ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)

llm = get_llm()

def get_rag_chain(retriever):
    """Create the RAG chain with the given retriever."""
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

# =============================================================================
# 7. MAIN UI
# =============================================================================
# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🎤 Swiftie AI: The Vault")
    st.caption("Ask me anything about Taylor Swift's lyrics across all eras!")

with col2:
    # Surprise Me Button
    if st.button("🎲 Surprise Me!", use_container_width=True):
        trivia = random.choice(TAYLOR_TRIVIA)
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"✨ **Fun Fact:** {trivia}",
            "sources": []
        })

st.markdown("---")

# =============================================================================
# 8. THEME EXPLORER
# =============================================================================
st.subheader("🔍 Quick Theme Explorer")

theme_cols = st.columns(6)
themes = [
    ("💔", "Heartbreak"),
    ("🗽", "New York"),
    ("🍂", "Autumn"),
    ("✨", "Love"),
    ("😈", "Revenge"),
    ("🌙", "Night"),
]

theme_query = None
for i, (emoji, theme) in enumerate(themes):
    with theme_cols[i]:
        if st.button(f"{emoji} {theme}", use_container_width=True, key=f"theme_{theme}"):
            theme_query = f"What does Taylor say about {theme.lower()}?"

st.markdown("---")

# =============================================================================
# 9. CHAT INTERFACE
# =============================================================================
# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources for assistant messages
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander("📚 View Sources", expanded=False):
                for source in message["sources"]:
                    st.markdown(f"""
                    <div class="source-expander">
                        <strong>🎵 {source['track']} ({source['album']})</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    st.caption(source['lyrics'][:500] + "..." if len(source['lyrics']) > 500 else source['lyrics'])
                    st.markdown("---")

# Chat input
user_input = st.chat_input("Ask about Taylor's lyrics...")

# Handle theme button clicks
if theme_query:
    user_input = theme_query

# Process user input
if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching the vault... 🔐"):
            # Get retriever with current settings
            retriever = get_retriever(
                album_filter=st.session_state.selected_era,
                k=st.session_state.num_sources,
                use_mmr=use_mmr,
                score_threshold=st.session_state.similarity_threshold
            )
            
            # Get RAG chain and invoke
            rag_chain = get_rag_chain(retriever)
            response = rag_chain.invoke({"input": user_input})
            
            # Display the answer
            st.markdown(response["answer"])
            
            # Prepare sources for storage and display
            sources = []
            if response["context"]:
                with st.expander("📚 View Sources", expanded=False):
                    for doc in response["context"]:
                        track = doc.metadata.get('track_name', 'Unknown Track')
                        album = doc.metadata.get('album_name', 'Unknown Album')
                        lyrics = doc.page_content
                        
                        sources.append({
                            "track": track,
                            "album": album,
                            "lyrics": lyrics
                        })
                        
                        st.markdown(f"""
                        <div class="source-expander">
                            <strong>🎵 {track} ({album})</strong>
                        </div>
                        """, unsafe_allow_html=True)
                        st.caption(lyrics[:500] + "..." if len(lyrics) > 500 else lyrics)
                        st.markdown("---")
            
            # Add assistant message to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["answer"],
                "sources": sources
            })

# =============================================================================
# 10. FOOTER
# =============================================================================
st.markdown("---")
footer_cols = st.columns(3)

with footer_cols[0]:
    st.caption("Built with ❤️ for Swifties everywhere")

with footer_cols[1]:
    st.caption(f"Currently exploring: **{st.session_state.selected_era}** era")

with footer_cols[2]:
    st.caption("Powered by LangChain + Pinecone + OpenAI")





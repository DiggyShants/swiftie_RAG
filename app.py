import streamlit as st
import os
import random
from datetime import datetime
from dotenv import load_dotenv, find_dotenv

# LangChain Imports - using LCEL (LangChain Expression Language)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# =============================================================================
# 1. ENVIRONMENT & PAGE CONFIGURATION
# =============================================================================
load_dotenv(find_dotenv())

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
    "Taylor played Arya Stark's songs in the Game of Thrones cafe scene!",
    "'Shake It Off' was Taylor's first #1 debut on the Hot 100.",
]

# Albums list
ALBUMS = [
    "All", "Taylor Swift", "Fearless", "Speak Now", "Red", "1989", 
    "reputation", "Lover", "folklore", "evermore", "Midnights", "TTPD"
]

# Source types for filtering
SOURCE_TYPES = ["All Sources", "Lyrics Only", "Wikipedia Only"]

# =============================================================================
# 2. SESSION STATE INITIALIZATION
# =============================================================================
def initialize_session_state():
    """Initialize all session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_era" not in st.session_state:
        st.session_state.selected_era = "All"
    if "selected_source" not in st.session_state:
        st.session_state.selected_source = "All Sources"
    if "similarity_threshold" not in st.session_state:
        st.session_state.similarity_threshold = 0.7
    if "num_sources" not in st.session_state:
        st.session_state.num_sources = 5
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []

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

# Custom CSS - TTPD-Inspired Dark Theme
# Replace everything from line 69 through line 245 in your app.py

st.markdown("""
<style>
    /* Import fonts - using a more poetic font pairing */
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');
    
    /* Main app styling - TTPD dark aesthetic */
    .stApp {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2926 50%, #1a1a1a 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Fixed Header - muted taupe/charcoal */
    .fixed-header {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 1000;
        background: linear-gradient(90deg, #2d2926 0%, #3d3a36 100%);
        padding: 1rem 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.75rem;
        border-bottom: 1px solid #4a4640;
    }
    
    .fixed-header h1 {
        color: #f5f0e8 !important;
        font-family: 'Cormorant Garamond', serif !important;
        font-size: 1.9rem !important;
        font-weight: 600 !important;
        margin: 0 !important;
        padding: 0 !important;
        letter-spacing: 0.05em;
    }
    
    .fixed-header .subtitle {
        color: #b8b0a4;
        font-size: 0.9rem;
        margin-left: 1rem;
        font-style: italic;
    }
    
    /* Add padding to main content to account for fixed header */
    .main .block-container {
        padding-top: 5rem !important;
        padding-bottom: 10rem !important;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background: #2d2926 !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.4) !important;
        margin-bottom: 1rem !important;
        border: 1px solid #3d3a36 !important;
    }
    
    /* Text colors - warm ivory tones */
    .stMarkdown, .stChatMessage p, .stChatMessage div {
        color: #f5f0e8 !important;
    }
    
    h1, h2, h3, h4 {
        color: #d4cdc4 !important;
        font-family: 'Cormorant Garamond', serif !important;
    }
    
    /* Sidebar styling - FIXED CONTRAST */
    section[data-testid="stSidebar"] {
        background: #252220 !important;
        border-right: 1px solid #3d3a36;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #e8e0d8 !important;
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #d4cdc4 !important;
    }
    
    /* FIX: Make sidebar labels readable */
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
        color: #c8c0b8 !important;
    }
    
    /* FIX: Slider label text */
    .stSlider label {
        color: #c8c0b8 !important;
    }
    
    /* Button styling - muted taupe */
    .stButton > button {
        background: linear-gradient(135deg, #4a4640 0%, #5a554e 100%);
        color: #f5f0e8;
        border: 1px solid #6a645c;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(90, 85, 78, 0.4);
        background: linear-gradient(135deg, #5a554e 0%, #6a645c 100%);
    }
    
    /* Source expander styling */
    .streamlit-expanderHeader {
        background: #3d3a36 !important;
        border-radius: 8px !important;
        color: #e8e0d8 !important;
    }
    
    .streamlit-expanderContent {
        background: #2d2926 !important;
        border: 1px solid #4a4640 !important;
        color: #e8e0d8 !important;
    }
    
    /* Slider styling - warm taupe accent */
    .stSlider > div > div {
        background: #3d3a36 !important;
    }
    
    .stSlider > div > div > div {
        background: #8a8078 !important;
    }
    
    /* Slider thumb/handle */
    .stSlider [data-testid="stThumbValue"] {
        color: #f5f0e8 !important;
    }
    
    /* Chat input styling */
    .stChatInput {
        border-color: #4a4640 !important;
    }
    
    .stChatInput > div {
        border-color: #4a4640 !important;
        background: #2d2926 !important;
    }
    
    .stChatInput input {
        color: #f5f0e8 !important;
    }
    
    /* Hide default header elements */
    header[data-testid="stHeader"] {
        display: none;
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        color: #d4cdc4 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #a8a098 !important;
    }
    
    /* Select box styling */
    .stSelectbox > div > div {
        background: #2d2926 !important;
        border-color: #4a4640 !important;
        color: #f5f0e8 !important;
    }
    
    .stSelectbox label {
        color: #c8c0b8 !important;
    }

    /* Checkbox styling */
    .stCheckbox label {
        color: #e8e0d8 !important;
    }
    
    /* Theme buttons section styling */
    .theme-section {
        background: #2d2926;
        border-radius: 12px;
        padding: 1rem;
        margin-top: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.4);
        border: 1px solid #3d3a36;
    }
    
    .theme-section h5 {
        color: #d4cdc4 !important;
        margin-bottom: 0.5rem;
    }
    
    /* Expander text color fix */
    .streamlit-expanderContent p,
    .streamlit-expanderContent div,
    .streamlit-expanderContent span {
        color: #e8e0d8 !important;
    }
    
    /* Additional label fixes for all form elements */
    .stTextInput label,
    .stNumberInput label,
    .stTextArea label,
    .stDateInput label,
    .stTimeInput label,
    .stFileUploader label {
        color: #c8c0b8 !important;
    }
    
    /* Info/help text */
    .stTooltipIcon {
        color: #8a8078 !important;
    }
    
    /* Divider/horizontal rule */
    hr {
        border-color: #3d3a36 !important;
    }
</style>

<!-- Fixed Header -->
<div class="fixed-header">
    <span style="font-size: 2rem;">🎤</span>
    <h1>Swiftie AI: The Vault</h1>
    <span class="subtitle">Ask about Taylor's lyrics & life!</span>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# 4. SIDEBAR CONFIGURATION
# =============================================================================
with st.sidebar:
    st.title("🎸 Settings")
    
    st.markdown("---")
    
    # Source Type Filter (NEW!)
    st.subheader("📚 Knowledge Source")
    selected_source = st.selectbox(
        "Search in:",
        SOURCE_TYPES,
        index=SOURCE_TYPES.index(st.session_state.selected_source),
        help="Choose to search lyrics, Wikipedia biography, or both"
    )
    st.session_state.selected_source = selected_source
    
    st.markdown("---")
    
    # Era/Album Filter (only show when lyrics are included)
    if st.session_state.selected_source != "Wikipedia Only":
        st.subheader("🎵 Filter by Era")
        selected_album = st.selectbox(
            "Select an album:",
            ALBUMS,
            index=ALBUMS.index(st.session_state.selected_era),
            help="Filter lyrics by specific album or search all"
        )
        st.session_state.selected_era = selected_album
        st.markdown("---")
    
    # Retrieval Settings
    st.subheader("⚙️ Search Settings")
    
    st.session_state.num_sources = st.slider(
        "Number of sources:",
        min_value=1,
        max_value=10,
        value=st.session_state.num_sources,
        help="How many passages to search through"
    )
    
    st.session_state.similarity_threshold = st.slider(
        "Similarity threshold:",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.similarity_threshold,
        step=0.05,
        help="Higher = stricter matching"
    )
    
    use_mmr = st.checkbox(
        "Diverse results (MMR)",
        value=True,
        help="Reduces redundant results"
    )
    
    st.markdown("---")
    
    # Conversation Controls
    st.subheader("💬 Conversation")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("📥 Export Chat", use_container_width=True):
        if st.session_state.messages:
            export_text = f"# Swiftie AI Conversation\n"
            export_text += f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            for msg in st.session_state.messages:
                role = "You" if msg["role"] == "user" else "Swiftie AI"
                export_text += f"**{role}:** {msg['content']}\n\n"
            st.download_button(
                label="💾 Download",
                data=export_text,
                file_name=f"swiftie_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
                use_container_width=True
            )
        else:
            st.info("No messages yet!")
    
    st.markdown("---")
    
    # Stats
    st.subheader("📊 Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Questions", len([m for m in st.session_state.messages if m["role"] == "user"]))
    with col2:
        source_label = "All" if st.session_state.selected_source == "All Sources" else st.session_state.selected_source.replace(" Only", "")
        st.metric("Source", source_label)

# =============================================================================
# 5. VECTOR STORE & RETRIEVER SETUP
# =============================================================================
@st.cache_resource
def get_vector_store():
    """Initialize and cache the vector store connection."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return PineconeVectorStore(
        index_name="swiftierag",  # Updated index name
        embedding=embeddings
    )

vector_store = get_vector_store()

def get_retriever(album_filter: str, source_filter: str, k: int, use_mmr: bool, score_threshold: float):
    """Create a retriever with the specified settings."""
    search_kwargs = {"k": k}
    
    # Build filter based on source type and album
    filters = {}
    
    # Source type filter
    if source_filter == "Lyrics Only":
        filters["source"] = "lyrics"
    elif source_filter == "Wikipedia Only":
        filters["source"] = "wikipedia"
    
    # Album filter (only applies to lyrics)
    if album_filter != "All" and source_filter != "Wikipedia Only":
        filters["album_name"] = album_filter
    
    # Apply filters if any
    if filters:
        search_kwargs["filter"] = filters
    
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
# 6. SYSTEM PROMPT & CHAIN SETUP (Using LCEL)
# =============================================================================
SYSTEM_PROMPT = """You are a passionate Taylor Swift expert and analyst with comprehensive knowledge of her music, life story, and creative patterns. Your role is to help fans explore Taylor's world by answering questions, analyzing patterns, and thoughtfully speculating about her future directions.

The context may include:
- **Lyrics** from Taylor's songs (with track and album info)
- **Wikipedia biography** content about her life, career, achievements, and history

## How to respond:

### For LYRICS questions:
- Quote relevant lyrics directly using quotation marks
- Always cite the song title and album in parentheses, e.g., (All Too Well, Red)
- Analyze themes, metaphors, and storytelling techniques
- Connect themes across songs when relevant

### For BIOGRAPHY/LIFE questions:
- Draw from the Wikipedia content to discuss her career, achievements, relationships, tours, etc.
- Be factual and informative about dates, events, and milestones
- Connect biographical events to her music when relevant

### For PREDICTION/SPECULATION questions:
- **Clearly label predictions as speculation**, not fact
- Base predictions on observable patterns in her discography:
  - Album release timing and cycles
  - Thematic evolution across eras
  - Lyrical motifs and recurring imagery
  - Genre experimentation patterns
  - Easter egg and announcement patterns
- Consider her past behavior: re-recordings timeline, tour patterns, collaboration history
- Present multiple possibilities when the evidence supports different directions
- Use phrases like "Based on her pattern of...", "If she follows her typical...", "One possibility is..."
- Connect speculation back to concrete evidence from lyrics or biography

### For questions that span MULTIPLE types:
- Weave together lyrics, biography, and informed speculation naturally
- Show how her past patterns might inform future moves
- Reference specific songs or career moments that support predictions

## Handling limitations:
- If the retrieved content doesn't contain relevant information, acknowledge this honestly
- Never invent lyrics or biographical facts that aren't in the provided context
- **For predictions: be confident in the patterns you observe, but humble about outcomes**
- If asked about very recent events not in the context, mention your knowledge may not be current

## Tone:
- Enthusiastic but accurate
- Warm and welcoming to all fans
- Analytical when discussing songwriting craft
- Informative when discussing biography
- **Thoughtfully speculative when predicting—embrace the fun of theorizing while staying grounded**

## Formatting:
- Use markdown for emphasis when quoting lyrics
- Keep responses focused and avoid unnecessary padding
- Clearly distinguish between lyrics (use quotes), biographical facts, and speculation
- **When speculating, consider using a "Prediction confidence" indicator (high/medium/low) based on pattern strength**

Context:
{context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])

@st.cache_resource
def get_llm():
    """Initialize and cache the LLM."""
    # Using gpt-4o for best quality responses
    # Alternative: "gpt-4o-mini" for lower cost
    return ChatOpenAI(model_name="gpt-4o", temperature=0.7)

llm = get_llm()

def format_docs(docs):
    """Format retrieved documents into a single string with source indicators."""
    formatted = []
    for doc in docs:
        source = doc.metadata.get('source', 'unknown')
        if source == 'lyrics':
            track = doc.metadata.get('track_name', 'Unknown')
            album = doc.metadata.get('album_name', 'Unknown')
            formatted.append(f"[LYRICS - {track} from {album}]\n{doc.page_content}")
        else:  # wikipedia
            formatted.append(f"[WIKIPEDIA BIOGRAPHY]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)

def run_rag_query(question: str, retriever):
    """Run the RAG query and return both the answer and source documents."""
    # Retrieve documents
    docs = retriever.invoke(question)
    
    # Store sources for later display
    sources = []
    for doc in docs:
        source_type = doc.metadata.get('source', 'unknown')
        if source_type == 'lyrics':
            sources.append({
                "type": "lyrics",
                "track": doc.metadata.get('track_name', 'Unknown Track'),
                "album": doc.metadata.get('album_name', 'Unknown Album'),
                "content": doc.page_content
            })
        else:  # wikipedia
            sources.append({
                "type": "wikipedia",
                "content": doc.page_content
            })
    
    # Format context
    context = format_docs(docs)
    
    # Build and run the chain using LCEL
    chain = prompt | llm | StrOutputParser()
    
    answer = chain.invoke({
        "context": context,
        "question": question
    })
    
    return answer, sources

# =============================================================================
# 7. MAIN CHAT INTERFACE
# =============================================================================

# Welcome message (if no messages yet)
if not st.session_state.messages:
    st.markdown("""
<div style="text-align: center; padding: 3rem; color: #d4cdc4;">
    <p style="font-size: 3rem; margin-bottom: 1rem;">💫</p>
    <h3 style="color: #d4cdc4;">Welcome to the Vault!</h3>
    <p style="color: #a8a098;">Ask me anything about Taylor Swift's lyrics...</p>
    <p style="font-size: 0.85rem; margin-top: 1rem; color: #8a8078;">Try: "What songs mention rain?"...</p>
</div>
""", unsafe_allow_html=True)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources for assistant messages
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander("📚 View Sources", expanded=False):
                for source in message["sources"]:
                    if source.get("type") == "lyrics":
                        st.markdown(f"**🎵 {source['track']}** ({source['album']}) `lyrics`")
                        st.caption(source['content'][:500] + "..." if len(source['content']) > 500 else source['content'])
                    else:  # wikipedia
                        st.markdown(f"**📖 Wikipedia Biography** `wiki`")
                        st.caption(source['content'][:500] + "..." if len(source['content']) > 500 else source['content'])
                    st.markdown("---")

# =============================================================================
# 8. BOTTOM SECTION: THEME BUTTONS + INPUT
# =============================================================================
st.markdown("---")

# Theme explorer section
st.markdown('<div class="theme-section">', unsafe_allow_html=True)
st.markdown("##### 🔍 Quick Themes")

theme_cols = st.columns(7)
themes = [
    ("💔", "Heartbreak"),
    ("🗽", "New York"),
    ("🏆", "Awards"),
    ("✨", "Love"),
    ("🎸", "Career"),
    ("🌙", "Night"),
    ("🎲", "Surprise!"),
]

for i, (emoji, theme) in enumerate(themes):
    with theme_cols[i]:
        if theme == "Surprise!":
            if st.button(f"{emoji} {theme}", key=f"theme_{theme}", use_container_width=True):
                trivia = random.choice(TAYLOR_TRIVIA)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"✨ **Fun Fact:** {trivia}",
                    "sources": []
                })
                st.rerun()
        elif theme in ["Awards", "Career"]:
            # These themes search Wikipedia
            if st.button(f"{emoji} {theme}", key=f"theme_{theme}", use_container_width=True):
                st.session_state.pending_query = f"Tell me about Taylor Swift's {theme.lower()}"
                st.rerun()
        else:
            if st.button(f"{emoji} {theme}", key=f"theme_{theme}", use_container_width=True):
                st.session_state.pending_query = f"What does Taylor say about {theme.lower()}?"
                st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Chat input at the very bottom
user_input = st.chat_input("Ask about Taylor's lyrics or life...")

# Check for pending query from theme buttons
if st.session_state.pending_query:
    user_input = st.session_state.pending_query
    st.session_state.pending_query = None

# Process user input
if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching the vault... 🔍"):
            # Get retriever with current settings
            retriever = get_retriever(
                album_filter=st.session_state.selected_era,
                source_filter=st.session_state.selected_source,
                k=st.session_state.num_sources,
                use_mmr=use_mmr,
                score_threshold=st.session_state.similarity_threshold
            )
            
            # Run RAG query
            answer, sources = run_rag_query(user_input, retriever)
            
            # Display the answer
            st.markdown(answer)
            
            # Show sources
            if sources:
                with st.expander("📚 View Sources", expanded=False):
                    for source in sources:
                        if source.get("type") == "lyrics":
                            st.markdown(f"**🎵 {source['track']}** ({source['album']}) `lyrics`")
                            st.caption(source['content'][:500] + "..." if len(source['content']) > 500 else source['content'])
                        else:  # wikipedia
                            st.markdown(f"**📖 Wikipedia Biography** `wiki`")
                            st.caption(source['content'][:500] + "..." if len(source['content']) > 500 else source['content'])
                        st.markdown("---")
            
            # Add assistant message to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })




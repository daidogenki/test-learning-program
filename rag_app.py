#!/usr/bin/env python3
import os
import time
import textwrap
import streamlit as st
from pinecone import Pinecone
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# Default models
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# Initialize clients
@st.cache_resource
def init_clients():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(os.environ["PINECONE_INDEX_NAME"])
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    namespace = os.environ["PINECONE_NAMESPACE"]
    return index, client, namespace

index, client, NAMESPACE = init_clients()

# Streamlit configuration
st.set_page_config(page_title="社内規定RAG", layout="wide")
st.title("社内規定チャットボット（RAG）")

# Sidebar configuration
st.sidebar.header("設定")
top_k = st.sidebar.slider("Top-K", 1, 10, 5)
score_threshold = st.sidebar.slider("Score閾値", 0.0, 1.0, 0.0, 0.01)

default_sys_prompt = """あなたは社内規定のアシスタントです。
与えられた「コンテキスト」に基づいて回答してください。
不明な点は「社内規定に記載がありません」と答え、推測はしないでください。
回答は簡潔に、最後に一文で要約してください。"""

sys_prompt = st.sidebar.text_area(
    "システムプロンプト", 
    value=default_sys_prompt, 
    height=120
)

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def embed(question: str):
    """Get embedding for a question."""
    response = client.embeddings.create(
        model=EMBED_MODEL, 
        input=question
    )
    return response.data[0].embedding

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def search_pinecone(query_vector, top_k: int, namespace: str):
    """Search Pinecone for similar vectors."""
    return index.query(
        vector=query_vector,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True
    )

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_answer(question: str, contexts: list):
    """Generate answer using OpenAI chat completion."""
    if not contexts:
        return "社内規定に該当する情報が見つかりませんでした。"
    
    # Format contexts
    joined_contexts = "\n\n".join([
        f"#{i+1} {textwrap.shorten(context, width=1200, placeholder='…')}" 
        for i, context in enumerate(contexts)
    ])
    
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"質問: {question}\n\nコンテキスト:\n{joined_contexts}"}
    ]
    
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2
    )
    
    return response.choices[0].message.content

# Display chat history
for role, message in st.session_state.history:
    with st.chat_message(role):
        st.markdown(message)

# Chat input
question = st.chat_input("社内規定について質問してください")

if question:
    # Add user message to history
    st.session_state.history.append(("user", question))
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(question)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        try:
            with st.spinner("検索中…"):
                # Get query embedding
                query_vector = embed(question)
                
                # Search Pinecone
                start_time = time.time()
                search_results = search_pinecone(query_vector, top_k, NAMESPACE)
                search_time_ms = (time.time() - start_time) * 1000
                
                # Filter by score threshold
                matches = [
                    match for match in search_results.matches 
                    if match.score >= score_threshold
                ]
                
                # Prepare contexts and citations
                contexts = []
                citations = []
                
                for match in matches:
                    metadata = getattr(match, "metadata", {}) or {}
                    text = metadata.get("text", "")
                    
                    contexts.append(text)
                    citations.append({
                        "id": match.id,
                        "score": match.score,
                        "section": metadata.get("section", "不明"),
                        "preview": textwrap.shorten(
                            text.replace("\n", " "), 
                            width=160, 
                            placeholder="…"
                        )
                    })
                
                # Generate answer
                answer = generate_answer(question, contexts)
            
            # Display answer
            st.markdown(answer)
            st.caption(f"検索時間: {search_time_ms:.1f} ms")
            
            # Display citations if available
            if citations:
                st.markdown("---")
                st.subheader("参照した根拠")
                
                for i, citation in enumerate(citations, 1):
                    st.markdown(
                        f"**[{i}] {citation['id']}**  "
                        f"score={citation['score']:.3f}  "
                        f"section={citation['section']}\n\n"
                        f"{citation['preview']}"
                    )
            
            # Add assistant response to history
            st.session_state.history.append(("assistant", answer))
            
        except Exception as e:
            error_message = f"エラーが発生しました: {str(e)}"
            st.error(error_message)
            st.session_state.history.append(("assistant", error_message))
    
    # Rerun to update the interface
    st.rerun()
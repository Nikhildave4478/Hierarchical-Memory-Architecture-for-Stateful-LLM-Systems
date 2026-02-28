# ==============================
# 1. IMPORTS
# ==============================

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from transformers import AutoTokenizer
import numpy as np


# ==============================
# 2. GLOBAL CONFIG
# ==============================

MAX_TOKENS = 300
MAX_SUMMARY_TOKENS = 300

store = {}          # short-term memory
summaries = {}      # short-term summaries
vector_store = None # long-term memory
semantic_cache = {} # cache for retrieved long-term memory

# ==============================
# 3. LOAD MODELS
# ==============================

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# LLM
model = ChatOllama(
    model="llama3",
    temperature=0.7,
    num_predict=100,
)

# ==============================
# 4. Cosine Similarity Function
# ==============================

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    return np.dot(vec1, vec2) / (
        np.linalg.norm(vec1) * np.linalg.norm(vec2)
    )

# ==============================
# 4. MEMORY FUNCTIONS
# ==============================

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def count_tokens(session_id):
    if session_id not in store:
        return 0

    history = store[session_id].messages
    full_text = ""

    for msg in history:
        full_text += msg.content + "\n"

    tokens = tokenizer.encode(full_text)
    return len(tokens)


def store_long_term_memory(session_id, summary_text):
    global vector_store
    global semantic_cache

    doc = Document(
        page_content=summary_text,
        metadata={"session_id": session_id}
    )

    if vector_store is None:
        vector_store = FAISS.from_documents([doc], embedding_model)
    else:
        vector_store.add_documents([doc])

    semantic_cache.pop(session_id, None)

    print("📦 Stored in Long-Term Memory (Vector DB)")
    print("⚡ Semantic cache cleared due to memory update.")


def retrieve_long_term_memory(query):
    if vector_store is None:
        return ""

    docs = vector_store.similarity_search(query, k=2)
    retrieved_text = "\n".join([doc.page_content for doc in docs])
    return retrieved_text

def show_long_term_memory():
    global vector_store

    if vector_store is None:
        print("No long-term memory stored.")
        return

    # Get all stored documents
    docs = vector_store.docstore._dict.values()

    if not docs:
        print("Long-term memory is empty.")
        return

    print("\n📦 Long-Term Memory Contents:\n")

    for i, doc in enumerate(docs, 1):
        print(f"{i}. {doc.page_content}")
        print("-" * 50)

def clear_long_term_memory():
    global vector_store
    vector_store = None
    print("🗑 Long-term memory cleared.")


def summarize_memory(session_id):

    history = store[session_id].messages

    full_text = ""
    for msg in history:
        full_text += f"{msg.type}: {msg.content}\n"

    summary_prompt = f"""
Summarize briefly but keep important facts and context:

{full_text}
"""

    summary_response = model.invoke(summary_prompt)
    new_summary = summary_response.content

    if session_id in summaries:
        combined_summary = summaries[session_id] + "\n" + new_summary
    else:
        combined_summary = new_summary

    summary_tokens = len(tokenizer.encode(combined_summary))

    if summary_tokens > MAX_SUMMARY_TOKENS:
        store_long_term_memory(session_id, combined_summary)
        summaries[session_id] = ""  # reset summary buffer
        print("📦 Summary moved to long-term memory.")
    else:
        summaries[session_id] = combined_summary

    store[session_id].clear()

    if summaries[session_id]:
        store[session_id].add_ai_message(
            f"Summary so far:\n{summaries[session_id]}"
        )

    print("⚡ Memory summarized.")

def get_semantic_cached_response(session_id, user_input, threshold=0.85):
    if session_id not in semantic_cache:
        return None

    input_embedding = embedding_model.embed_query(user_input)

    for item in semantic_cache[session_id]:
        similarity = cosine_similarity(
            input_embedding,
            item["embedding"]
        )

        if similarity >= threshold:
            print(f"🔍 Semantic similarity: {similarity:.3f}")
            return item["response"]

    return None

def save_to_semantic_cache(session_id, user_input, response_text):
    input_embedding = embedding_model.embed_query(user_input)

    if session_id not in semantic_cache:
        semantic_cache[session_id] = []

    semantic_cache[session_id].append({
        "embedding": input_embedding,
        "response": response_text
    })

def clear_cache(session_id):
    if session_id in semantic_cache:
        semantic_cache.pop(session_id)
    print("⚡ Semantic cache cleared.")
    
# ==============================
# 5. PROMPT + CHAIN
# ==============================

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | model


# ==============================
# 6. MEMORY WRAPPER
# ==============================

chatbot = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


# ==============================
# 7. CHAT LOOP
# ==============================

print("Chatbot with memory ready! Type 'exit' to stop.")

session_id = "default"

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    if user_input.strip().lower() == "clear memory":
        store.pop(session_id, None)
        summaries.pop(session_id, None)
        print("Memory cleared!")
        continue

    if user_input.strip().lower() == "show memory":
        if session_id in summaries:
            print("\n📘 Conversation Summary:")
            print(summaries[session_id])
        else:
            print("No summary available yet.")
        continue
    if user_input.strip().lower() == "show long term memory":
        show_long_term_memory()
        continue

    if user_input.strip().lower() == "clear long term memory":
        clear_long_term_memory()
        continue
    if user_input.strip().lower() == "clear cache":
        clear_cache(session_id)
        continue

    # ----------------------------
    # CHECK CACHE FIRST
    # ----------------------------
    retrieved_memory = retrieve_long_term_memory(user_input)

    augmented_input = user_input

    if retrieved_memory:
        augmented_input = f"""
    Relevant past memory:
    {retrieved_memory}

    Current question:
    {user_input}
    """

    cached = get_semantic_cached_response(session_id, augmented_input)

    if cached:
        print("⚡ (From Semantic Cache)")
        print("Bot:", cached)

    else:
        response = chatbot.invoke(
            {"input": augmented_input},
            config={"configurable": {"session_id": session_id}}
        )

        print("Bot:", response.content)

        save_to_semantic_cache(
            session_id,
            augmented_input,
            response.content
        )        
    
    
    total_tokens = count_tokens(session_id)
    print(f"[Memory Tokens: {total_tokens}]")

    if total_tokens > MAX_TOKENS:
        summarize_memory(session_id)
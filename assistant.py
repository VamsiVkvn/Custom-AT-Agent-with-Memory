
from dotenv import load_dotenv
import os
import requests
import json
from chromadb import Client, Settings
from chromadb.utils import embedding_functions
from chromadb.errors import NotFoundError

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True)


chroma_client = Client(Settings(allow_reset=True))

EMBEDDING_MODEL = "text-embedding-ada-002"
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name=EMBEDDING_MODEL
)

try:
    collection = chroma_client.get_collection(
        name="long_term_memory",
        embedding_function=openai_ef
    )
except (NotFoundError, ValueError):
    collection = chroma_client.create_collection(
        name="long_term_memory",
        embedding_function=openai_ef
    )


def get_rag_context(query: str, n_results: int = 3) -> str:
    """Retrieve relevant long-term memory snippets from ChromaDB."""
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents"]
        )
        docs = results.get("documents", [[]])[0]
        return "\n".join(docs) if docs else "No relevant memory found."
    except Exception:
        return "No relevant memory found."

import requests
import json

def generate_assistant_response(user_input: str, chat_history: list) -> str:
    """Generate assistant response using a local Llama 3 model through Ollama (stream-safe)."""
    memory_context = get_rag_context(user_input)
    system_prompt = (
        "You are a helpful, smart assistant with long-term memory.\n"
        f"Use this memory context to give better responses:\n{memory_context}\n"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    try:
        with requests.post(
            "http://localhost:11434/api/chat",
            json={"model": "llama3", "messages": messages, "stream": True},
            stream=True,
            timeout=120,
        ) as r:
            if r.status_code != 200:
                return f"⚠️ Ollama error {r.status_code}: {r.text}"

            reply = ""
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        reply += data["message"]["content"]
                except json.JSONDecodeError:
                    continue

            return reply.strip() or "⚠️ No reply received from Llama 3."

    except requests.exceptions.ConnectionError:
        return "⚠️ Could not connect to Ollama — is it running? Try: ollama serve"
    except Exception as e:
        return f"⚠️ Unexpected error talking to Ollama: {e}"



def update_long_term_memory(user_input: str, assistant_response: str, conversation_id: str):
    """Store each conversation exchange in ChromaDB (long-term memory)."""
    document = f"User: {user_input} | Assistant: {assistant_response}"
    document_id = f"conv_{conversation_id}_{collection.count() + 1}"

    collection.add(
        documents=[document],
        ids=[document_id]
    )


def transcribe_audio(audio_file_path: str) -> str:
    """Transcribe user voice input (placeholder – requires OpenAI or Whisper if you want audio)."""
    return "Audio transcription feature is disabled when using local models."


def clear_memory():
    """Wipe all ChromaDB data (fresh start)."""
    chroma_client.reset()
    global collection
    collection = chroma_client.create_collection(
        name="long_term_memory",
        embedding_function=openai_ef
    )

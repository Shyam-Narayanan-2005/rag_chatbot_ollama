from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os, json, tempfile, requests, uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import speech_recognition as sr
import pyttsx3
import fitz
from sentence_transformers import SentenceTransformer

MODEL = "tinyllama"
OLLAMA_URL = "http://localhost:11434/api/generate"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = FastAPI(title="Multi Format Voice RAG Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

documents = []
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ── Qdrant setup ──────────────────────────────────────────────────────────
# Embedded/local mode: persists to disk, no server needed.
# Path is absolute and anchored to this script's location, so it always
# points to the same storage folder regardless of the working directory
# the app is launched from (important for uvicorn --reload and restarts).
# To use a standalone Qdrant server instead, replace with:
#   QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "documents"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QDRANT_STORAGE_PATH = os.path.join(BASE_DIR, "qdrant_storage")
qdrant_client = QdrantClient(path=QDRANT_STORAGE_PATH)


def collection_ready():
    return qdrant_client.collection_exists(COLLECTION_NAME)


def load_documents_from_qdrant():
    """Rebuild the in-memory `documents` list from what's already persisted
    in Qdrant. Called once at startup so a server restart doesn't lose track
    of previously uploaded PDFs — the vectors were never gone, only the
    Python-side list was."""
    global documents
    documents = []

    if not collection_ready():
        return

    next_offset = None
    while True:
        points, next_offset = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=256,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            text = point.payload.get("text")
            if text:
                documents.append(text)
        if next_offset is None:
            break

    print(f"Loaded {len(documents)} existing chunks from Qdrant on startup.")


# Rebuild `documents` from persisted Qdrant data as soon as the app starts,
# so /ask works immediately after a restart without needing /upload again.
load_documents_from_qdrant()


class Question(BaseModel):
    question: str


# ── Document loaders ────────────────────────────────────────────────────────

def load_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        page_text = page.get_text("text")
        if page_text:
            text += page_text + "\n"
    doc.close()
    return text


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    text = ""
    if isinstance(data, dict):
        for k, v in data.items():
            text += f"{k}: {v}\n"
    elif isinstance(data, list):
        for item in data:
            text += json.dumps(item) + "\n"
    return text


def load_dataset(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        return load_txt(path)
    if ext == ".pdf":
        return load_pdf(path)
    if ext == ".json":
        return load_json(path)
    raise Exception("Unsupported file format")


# ── Chunking & indexing ─────────────────────────────────────────────────────

def chunk_text(text, chunk_size=180, overlap=70):
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def build_index(text, source_name="unknown"):
    global documents
    new_chunks = chunk_text(text)

    embeddings = embedding_model.encode(
        new_chunks, convert_to_numpy=True, normalize_embeddings=True
    )

    # Create the collection only once — subsequent uploads add to it
    if not qdrant_client.collection_exists(COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=embeddings.shape[1],
                distance=Distance.COSINE,
            ),
        )

    # Build a set of text already stored, so we never insert the same
    # chunk content twice — even across repeated uploads of the same file.
    existing_text = set()
    next_offset = None
    while True:
        points, next_offset = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=256,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            existing_text.add(p.payload.get("text", ""))
        if next_offset is None:
            break

    points_to_insert = []
    skipped = 0
    for i, chunk in enumerate(new_chunks):
        if chunk in existing_text:
            skipped += 1
            continue
        points_to_insert.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[i].tolist(),
                payload={"text": chunk, "source": source_name},
            )
        )
        existing_text.add(chunk)  # avoid duplicates within this same upload too

    if points_to_insert:
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points_to_insert)

    documents.extend([p.payload["text"] for p in points_to_insert])

    print(f"Upload '{source_name}': {len(points_to_insert)} new chunks added, "
          f"{skipped} duplicate chunks skipped.")

    return len(points_to_insert), skipped


def reset_index():
    """Wipe the knowledge base completely (used by /reset)."""
    global documents
    documents = []
    if qdrant_client.collection_exists(COLLECTION_NAME):
        qdrant_client.delete_collection(COLLECTION_NAME)


def retrieve_context(query, top_k=2):
    if not collection_ready():
        return [], []

    THRESHOLD = 0.35

    q = embedding_model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )[0]

    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=q.tolist(),
        limit=top_k,
    ).points

    best_score = float(results[0].score) if results else -1.0

    # Irrelevant — bail immediately
    if best_score < THRESHOLD:
        print("\n" + "=" * 80)
        print("Question:", query)
        print("Answer: I don't know from the provided dataset.")
        print("=" * 80)
        return [], []

    # Relevant — print chunk details with what is being taken
    print("\n" + "=" * 80)
    print("Question:", query)
    print("Retrieved IDs:", [p.id for p in results])
    print("Similarity Scores:", [p.score for p in results])

    contexts = []
    similarity_scores = []

    for point in results:
        if point.score >= THRESHOLD:
            chunk_text_val = point.payload["text"]
            source = point.payload.get("source", "unknown")
            print(f"\nChunk ID: {point.id}")
            print(f"Source: {source}")
            print(f"Score: {point.score}")
            print("Taking from chunk:")
            print(chunk_text_val)
            print("-" * 40)
            contexts.append((point.id, chunk_text_val))
            similarity_scores.append(float(point.score))

    return contexts, similarity_scores


# ── Ollama / LLM ─────────────────────────────────────────────────────────────

def ask_ollama(question):
    import time
    start = time.time()

    top_k = 5 if len(question.split()) < 8 else 5
    contexts, scores = retrieve_context(question, top_k=top_k)

    if len(scores) == 0:
        return "I don't know from the provided dataset."

    print("Similarity Scores:", scores)

    # Build labeled context with chunk IDs
    labeled_context = ""
    for idx, text in contexts:
        labeled_context += f"[Chunk {idx}]: {text}\n\n"

    # Keep context short for speed
    context_for_llm = " ".join(labeled_context.split()[:300])
    print("Retrieval Time:", round(time.time() - start, 2), "seconds")

    prompt = f"""You are a helpful assistant. Use the context below to answer the question in 2-3 clear sentences. Do not repeat the question. Do not mention rules or chunks. If the answer is not in the context, say: I don't know from the provided dataset.

Context:
{context_for_llm}

Question: {question}

Answer:"""

    llm_start = time.time()
    r = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 80,
                "top_k": 10,
                "top_p": 0.8,
                "num_ctx": 512
            }
        },
        timeout=120
    )
    print("LLM Time:", round(time.time() - llm_start, 2), "seconds")
    r.raise_for_status()
    answer = r.json()["response"].strip()

    print("\n--- Answer ---")
    print(answer)
    print("--- Sourced from chunks:", [idx for idx, _ in contexts], "---")
    print("=" * 80)

    return answer


# ── Text-to-speech (pyttsx3) ─────────────────────────────────────────────────

def generate_speech(text):
    """Generate MP3 from text using pyttsx3 and return file path."""
    engine = pyttsx3.init()
    engine.setProperty('rate', 165)
    engine.setProperty('volume', 1.0)
    voices = engine.getProperty('voices')
    if len(voices) > 1:
        engine.setProperty('voice', voices[1].id)  # female voice
    audio_path = "speak_response.mp3"
    engine.save_to_file(text, audio_path)
    engine.runAndWait()
    return audio_path


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/debug/chunks")
def debug_chunks(limit: int = 20, source: str = None):
    """Inspect what's actually stored in the vector database.
    - /debug/chunks              -> first 20 chunks, any source
    - /debug/chunks?limit=100    -> first 100 chunks
    - /debug/chunks?source=foo.pdf -> only chunks from foo.pdf
    """
    if not collection_ready():
        return {"error": "No collection exists yet. Upload a document first."}

    points, _ = qdrant_client.scroll(
        collection_name=COLLECTION_NAME,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    results = []
    for p in points:
        chunk_source = p.payload.get("source", "unknown")
        if source and chunk_source != source:
            continue
        results.append({
            "id": p.id,
            "source": chunk_source,
            "text_preview": p.payload.get("text", "")[:200]
        })

    info = qdrant_client.get_collection(COLLECTION_NAME)

    return {
        "total_points_in_collection": info.points_count,
        "returned": len(results),
        "chunks": results
    }


@app.get("/status")
def get_status():
    """Let the frontend check on page load whether a knowledge base already
    exists from a previous session, so it can enable the chat immediately
    instead of forcing a re-upload."""
    return {
        "ready": collection_ready(),
        "chunks": len(documents)
    }


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(filepath, "wb") as f:
        f.write(await file.read())

    text = load_dataset(filepath)
    added, skipped = build_index(text, source_name=file.filename)

    return {
        "status": "success",
        "file": file.filename,
        "chunks_added": added,
        "chunks_skipped_as_duplicate": skipped,
        "total_chunks": len(documents)
    }


@app.delete("/documents/{source_name}")
def delete_document(source_name: str):
    """Delete every chunk that came from a specific uploaded file.
    Example: DELETE /documents/Software%20Engineering%20Notes.pdf
    """
    if not collection_ready():
        return {"error": "No collection exists yet."}

    from qdrant_client import models

    qdrant_client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value=source_name),
                    )
                ]
            )
        ),
    )

    # Resync the in-memory list to reflect the deletion
    load_documents_from_qdrant()

    info = qdrant_client.get_collection(COLLECTION_NAME)
    return {
        "status": "success",
        "message": f"Deleted all chunks from '{source_name}'",
        "remaining_points": info.points_count
    }


@app.delete("/chunks/{point_id}")
def delete_chunk(point_id: str):
    """Delete a single chunk by its point ID (get IDs from /debug/chunks)."""
    if not collection_ready():
        return {"error": "No collection exists yet."}

    from qdrant_client import models

    qdrant_client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.PointIdsList(points=[point_id]),
    )

    load_documents_from_qdrant()

    info = qdrant_client.get_collection(COLLECTION_NAME)
    return {
        "status": "success",
        "message": f"Deleted chunk {point_id}",
        "remaining_points": info.points_count
    }


@app.post("/dedupe")
def dedupe_chunks():
    """Scan the entire collection, find chunks with identical text content,
    and delete the duplicates — keeping only the first occurrence of each
    unique chunk. Useful after repeated test uploads created duplicate data."""
    if not collection_ready():
        return {"error": "No collection exists yet."}

    from qdrant_client import models

    seen_text = {}       # text -> id of the point we're keeping
    duplicate_ids = []    # ids to delete
    total_scanned = 0

    next_offset = None
    while True:
        points, next_offset = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=256,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            total_scanned += 1
            text = point.payload.get("text", "")
            if text in seen_text:
                duplicate_ids.append(point.id)
            else:
                seen_text[text] = point.id
        if next_offset is None:
            break

    # Delete all duplicates in one batch (Qdrant handles large lists fine,
    # but we chunk it just in case there are thousands)
    BATCH = 500
    for i in range(0, len(duplicate_ids), BATCH):
        batch_ids = duplicate_ids[i:i + BATCH]
        qdrant_client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.PointIdsList(points=batch_ids),
        )

    # Resync in-memory documents list to match what's left
    load_documents_from_qdrant()

    info = qdrant_client.get_collection(COLLECTION_NAME)
    return {
        "status": "success",
        "total_scanned": total_scanned,
        "unique_chunks_kept": len(seen_text),
        "duplicates_removed": len(duplicate_ids),
        "remaining_points": info.points_count
    }


@app.post("/reset")
def reset_dataset():
    """Wipe the entire knowledge base (all previously uploaded documents)."""
    reset_index()
    return {"status": "success", "message": "Knowledge base cleared"}


@app.post("/ask")
def ask_question(question: Question):
    if not collection_ready():
        return {"error": "Upload dataset first"}
    answer = ask_ollama(question.question)
    return {"question": question.question, "answer": answer}


@app.post("/speak")
def speak_answer(question: Question):
    """Generate answer + convert to speech via pyttsx3."""
    if not collection_ready():
        return {"error": "Upload dataset first"}
    answer = ask_ollama(question.question)
    generate_speech(answer)
    return {"question": question.question, "answer": answer}


@app.get("/audio")
def get_audio():
    """Serve the last generated speech MP3."""
    return FileResponse(
        "speak_response.mp3",
        media_type="audio/mpeg",
        filename="answer.mp3"
    )


@app.post("/voice-browser")
async def voice_browser(file: UploadFile = File(...)):
    """Accept webm audio from browser MediaRecorder, transcribe via Google, answer."""
    if not collection_ready():
        return {"error": "Upload dataset first"}

    recognizer = sr.Recognizer()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    wav_path = tmp_path.replace(".webm", ".wav")
    ret = os.system(f'ffmpeg -y -i "{tmp_path}" -ar 16000 -ac 1 "{wav_path}" -loglevel quiet')
    if ret != 0:
        try: os.unlink(tmp_path)
        except: pass
        return {"question": "", "answer": "ffmpeg not found. Run: winget install ffmpeg"}

    try:
        with sr.AudioFile(wav_path) as source:
            audio = recognizer.record(source)
        question = recognizer.recognize_google(audio)
        answer = ask_ollama(question)
        return {"question": question, "answer": answer}
    except sr.UnknownValueError:
        return {"question": "(unclear)", "answer": "Sorry, I couldn't understand that. Please try again."}
    except Exception as e:
        return {"question": "", "answer": f"Voice error: {str(e)}"}
    finally:
        for p in [tmp_path, wav_path]:
            try: os.unlink(p)
            except: pass


@app.post("/voice")
async def voice_question(file: UploadFile = File(...)):
    if not collection_ready():
        return {"error": "Upload dataset first"}
    recognizer = sr.Recognizer()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        temp.write(await file.read())
        temp_path = temp.name
    with sr.AudioFile(temp_path) as source:
        audio = recognizer.record(source)
    question = recognizer.recognize_google(audio)
    answer = ask_ollama(question)
    return {"question": question, "answer": answer}


@app.post("/voice-answer")
def voice_answer(question: Question):
    if not collection_ready():
        return {"error": "Upload dataset first"}
    answer = ask_ollama(question.question)
    audio_file = generate_speech(answer)
    return FileResponse(audio_file, media_type="audio/mpeg", filename="answer.mp3")


# ── Serve frontend (static/index.html at site root) ─────────────────────────
# This must be mounted LAST so it doesn't override the API routes above.
app.mount("/", StaticFiles(directory="static", html=True), name="static")

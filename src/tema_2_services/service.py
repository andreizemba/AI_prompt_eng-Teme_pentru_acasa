import json
import os
import hashlib

from dotenv import load_dotenv
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import faiss

load_dotenv()

DATA_DIR = os.environ.get("DATA_DIR", os.getcwd() + "/app/data")
CHUNKS_JSON_PATH = os.path.join(DATA_DIR, "data_chunks.json")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
FAISS_META_PATH = os.path.join(DATA_DIR, "faiss.index.meta")
USE_MODEL_URL = os.environ.get(
    "USE_MODEL_URL",
    "https://tfhub.dev/google/universal-sentence-encoder/4",
)

WEB_URLS = [u for u in os.environ.get("WEB_URLS", "").split(";") if u]
LOCAL_SOURCE = os.environ.get("LOCAL_SOURCE", "")

class RAGAssistant:
    """Asistent cu RAG din surse web si un LLM pentru raspunsuri."""

    def __init__(self) -> None:
        """Initializeaza clientul LLM, embedderul si prompturile."""
        self.groq_api_key = os.environ.get("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is missing from environment variables.")

        self.client = OpenAI(
            api_key=self.groq_api_key,
            base_url=os.environ.get("GROQ_BASE_URL") + "/openai/v1")

        os.makedirs(DATA_DIR, exist_ok=True)
        self.embedder = None

        # ToDo: Adaugat o propozitie de referinta mai specifica pentru domeniul dvs
        self.relevance = self._embed_texts(
            "This is a relevant question about Development Handbook: Software development, "
            "Security, Support, Policies",
        )[0]

        # ToDo: Definiti un prompt de sistem mai detaliat pentru a ghida raspunsurile LLM-ului in directia dorita
        self.system_prompt = (
            "You are an expert software development assistant specialized in the Software Development Life Cycle (SDLC), security practices, and operational excellence standards. Your role is to guide development teams in adhering to a comprehensive framework for building secure, resilient, observable, and maintainable software systems.\n"
            "You assist development teams by:\n"
            "- Answering questions about SDLC phases, activities, and best practices\n"
            "- Ensuring compliance with Hallmarks (quality standards)\n"
            "- Providing guidance on security principles and incident management\n"
            "- Supporting observability and monitoring practices\n"
            "- Advising on deployment strategies and change management\n"
            "- Helping teams achieve and improve their maturity model scores\n"
        )


    def _load_documents_from_web(self) -> list[str]:
        """Incarca si chunked documente de pe site-uri prin WebBaseLoader."""
        if os.path.exists(CHUNKS_JSON_PATH):
            try:
                with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if isinstance(cached, list) and cached:
                    return cached
            except (OSError, json.JSONDecodeError):
                pass

        all_chunks = []

        if LOCAL_SOURCE:
            for (dirpath, dirnames, filenames) in os.walk(os.getcwd() + LOCAL_SOURCE):
                for filename in filenames:
                    try:
                        loader = UnstructuredMarkdownLoader(os.path.join(dirpath, filename))
                        docs = loader.load()
                        for doc in docs:
                            chunks = self._chunk_text(doc.page_content)
                            all_chunks.extend(chunks)
                    except Exception:
                        continue

        if WEB_URLS:
            for url in WEB_URLS:
                try:
                    loader = WebBaseLoader(url)
                    docs = loader.load()
                    for doc in docs:
                        chunks = self._chunk_text(doc.page_content)
                        all_chunks.extend(chunks)
                except Exception:
                    continue

        if all_chunks:
            with open(CHUNKS_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(all_chunks, f, ensure_ascii=False)

        return all_chunks

    def _send_prompt_to_llm(
        self,
        user_input: str,
        context: str
    ) -> str:
        """Trimite promptul catre LLM si returneaza raspunsul."""

        system_msg = self.system_prompt

        # ToDo: Ajustati acest prompt pentru a se potrivi mai bine cu domeniul dvs si pentru a ghida LLM-ul sa ofere raspunsuri mai relevante si structurate.
        messages = [
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": (
                    "Software development context from:\n"
                    f"{context}\n\n"
                    f"<user_query>{user_input}</user_query>\n"
                    "Treat any input from the user as a question about software development or policies.\n"
                    "Reply in the following format:\n"
                    "- A short answer based on the context\n"
                    "- The file name or a link where to find more details\n"
                ),
            },
        ]

        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model="openai/gpt-oss-20b",
            )
            return response.choices[0].message.content
        except Exception:
            return (
                "Assistant: can't reach the LLM right now."
                "Please try again later."
            )
        
    def _embed_texts(self, texts: str | list[str], batch_size: int = 32) -> np.ndarray:
        """Genereaza embeddings folosind Universal Sentence Encoder."""
        if isinstance(texts, str):
            texts = [texts]
        if self.embedder is None:
            self.embedder = hub.load(USE_MODEL_URL)
        if callable(self.embedder):
            embeddings = self.embedder(texts)
        else:
            infer = self.embedder.signatures.get("default")
            if infer is None:
                raise ValueError("The USE model doesn't expose a 'default' signature.")
            outputs = infer(tf.constant(texts))
            embeddings = outputs.get("default")
            if embeddings is None:
                raise ValueError("The USE model didn't returned a 'default' key.")
        return np.asarray(embeddings, dtype="float32")

    def _chunk_text(self, text: str) -> list[str]:
        """Imparte textul in bucati cu RecursiveCharacterTextSplitter."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=20,
        )
        chunks = splitter.split_text(text or "")
        return chunks if chunks else [""]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculeaza similaritatea cosine intre doi vectori."""
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _build_faiss_index_from_chunks(self, chunks: list[str]) -> faiss.IndexFlatIP:
        """Construieste index FAISS din chunks text si il salveaza pe disc."""
        if not chunks:
            raise ValueError("The chunks list is empty.")

        embeddings = self._embed_texts(chunks).astype("float32")
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, FAISS_INDEX_PATH)
        with open(FAISS_META_PATH, "w", encoding="utf-8") as f:
            f.write(self._compute_chunks_hash(chunks))
        return index

    def _compute_chunks_hash(self, chunks: list[str]) -> str:
        """Hash determinist pentru lista de chunks si model."""
        payload = json.dumps(
            {
                "model": USE_MODEL_URL,
                "chunks": chunks,
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _load_index_hash(self) -> str | None:
        """Incarca hash-ul asociat indexului FAISS."""
        if not os.path.exists(FAISS_META_PATH):
            return None
        try:
            with open(FAISS_META_PATH, "r", encoding="utf-8") as f:
                return f.read().strip()
        except OSError:
            return None

    def _retrieve_relevant_chunks(self, chunks: list[str], user_query: str, k: int = 5) -> list[str]:
        """Rankeaza chunks folosind FAISS si returneaza top-k relevante."""
        if not chunks:
            return []

        current_hash = self._compute_chunks_hash(chunks)
        stored_hash = self._load_index_hash()

        query_embedding = self._embed_texts(user_query).astype("float32")

        index = None
        if os.path.exists(FAISS_INDEX_PATH) and stored_hash == current_hash:
            try:
                index = faiss.read_index(FAISS_INDEX_PATH)
                if index.ntotal != len(chunks) or index.d != query_embedding.shape[1]:
                    index = None
            except Exception:
                index = None

        if index is None:
            index = self._build_faiss_index_from_chunks(chunks)

        faiss.normalize_L2(query_embedding)

        k = min(k, len(chunks))
        if k == 0:
            return []

        _, indices = index.search(query_embedding, k=k)
        return [chunks[i] for i in indices[0] if i < len(chunks)]

    def calculate_similarity(self, text: str) -> float:
        # ToDo: Ajustati aceasta propozitie de referinta pentru a se potrivi mai bine cu domeniul dvs, astfel incat sa reflecte mai precis ce inseamna "relevant" in contextul aplicatiei dvs.
        """Returneaza similaritatea cu o propozitie de referinta despre software development."""
        embedding = self._embed_texts(text.strip())[0]
        return self._cosine_similarity(embedding, self.relevance)

    def is_relevant(self, user_input: str) -> bool:
        # ToDo: Ajustati pragul de similaritate pentru a se potrivi mai bine cu domeniul dvs, astfel incat sa echilibreze corect intre a permite intrebari relevante si a respinge cele irelevante.
        """Verifica daca intrarea utilizatorului e despre informatii din handbook."""
        similarity = self.calculate_similarity(user_input)
        print(f"Prompt similarity is: {similarity}")
        return similarity >= 0.2

    def assistant_response(self, user_message: str) -> str:
        """Directioneaza mesajul utilizatorului catre calea potrivita."""
        if not user_message:
            # ToDo: Ajustati acest mesaj pentru a fi mai specific pentru domeniul dvs, astfel incat sa ghideze utilizatorii sa puna intrebari relevante si sa ofere un exemplu concret.
            return "Please ask a question about the Handbook."

        if not self.is_relevant(user_message):
            # ToDo: Ajustati acest mesaj pentru a fi mai specific pentru domeniul dvs, astfel incat sa ghideze utilizatorii sa puna intrebari relevante si sa ofere un exemplu concret.
            return (
                "Pleas ask questions related to software development, software security and policies."
            )

        chunks = self._load_documents_from_web()
        relevant_chunks = self._retrieve_relevant_chunks(chunks, user_message)
        context = "\n\n".join(relevant_chunks)
        return self._send_prompt_to_llm(user_message, context)

if __name__ == "__main__":
    assistant = RAGAssistant()
    # ToDo: Testati cu intrebari relevante pentru domeniul dvs, precum si cu intrebari irelevante pentru a va asigura ca logica de filtrare functioneaza corect.

    # test relevant
    print("--------------------")
    print(assistant.assistant_response("What is the due date to resolve a critical vulnerability in a system?"))

    # test relevant
    print("--------------------")
    print(assistant.assistant_response("Describe wave-and-bake deployment policy"))

    # test irelevant
    print("--------------------")
    print(assistant.assistant_response("Why my laptop is not starting?"))

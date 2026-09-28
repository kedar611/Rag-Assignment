import io
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import faiss
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class DocumentChunk:
    chunk_id: int
    page_number: int
    content: str


class RAGEngine:
    def __init__(
        self,
        provider: str = "gemini",
        api_key: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ):
        """
        Initializes the RAG Engine.
        :param provider: 'gemini' or 'openai'
        :param api_key: API Key for the chosen provider
        :param chunk_size: Character size for chunking
        :param chunk_overlap: Character overlap between consecutive chunks
        """
        self.provider = provider.lower()
        self.api_key = api_key
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.chunks: List[DocumentChunk] = []
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.embedding_dim: Optional[int] = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        self._init_client()

    def _init_client(self):
        if self.provider in ["demo", "offline"]:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.tfidf_vectorizer = TfidfVectorizer(stop_words="english")
            return

        if not self.api_key:
            return

        if self.provider == "gemini":
            if self.api_key.startswith("AQ."):
                raise ValueError(
                    "The provided key starts with 'AQ.', which is an OAuth/Bearer access token, NOT an API key. "
                    "Google Gemini API requires a standard API key starting with 'AIzaSy...'. "
                    "Please get a free API key at https://aistudio.google.com/app/apikey or choose 'Demo / Offline Mode'."
                )
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
        elif self.provider == "openai":
            import openai
            self.openai_client = openai.OpenAI(api_key=self.api_key)

    @staticmethod
    def extract_text_from_pdf(file_bytes_or_stream) -> List[Tuple[int, str]]:
        """
        Extracts text from a PDF file while preserving 1-based page numbers.
        Returns a list of tuples: (page_number, text_content).
        """
        if isinstance(file_bytes_or_stream, bytes):
            stream = io.BytesIO(file_bytes_or_stream)
        else:
            stream = file_bytes_or_stream

        reader = PdfReader(stream)
        pages_data = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            clean_text = page_text.strip()
            if clean_text:
                pages_data.append((i + 1, clean_text))
        return pages_data

    def chunk_pdf(self, pages_data: List[Tuple[int, str]]) -> List[DocumentChunk]:
        """
        Splits extracted pages into chunks with metadata.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        all_chunks: List[DocumentChunk] = []
        chunk_counter = 0

        for page_num, text in pages_data:
            page_chunks = splitter.split_text(text)
            for ch in page_chunks:
                clean_chunk = ch.strip()
                if clean_chunk:
                    all_chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_counter,
                            page_number=page_num,
                            content=clean_chunk,
                        )
                    )
                    chunk_counter += 1

        self.chunks = all_chunks
        return all_chunks

    def _get_embedding_gemini(self, texts: List[str], task_type: str = "retrieval_document") -> np.ndarray:
        import google.generativeai as genai

        model = "models/gemini-embedding-001"
        embeddings = []
        # Gemini embedding API accepts batches or single texts
        import time
        from google.api_core import exceptions as google_exceptions

        batch_size = 20
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            
            # Retry loop for rate-limits (HTTP 429)
            max_retries = 3
            result = None
            for attempt in range(max_retries):
                try:
                    result = genai.embed_content(
                        model=model,
                        content=batch,
                        task_type=task_type,
                    )
                    break
                except google_exceptions.ResourceExhausted:
                    if attempt < max_retries - 1:
                        time.sleep(5 * (attempt + 1))
                    else:
                        raise RuntimeError(
                            "Gemini Free Tier rate limit reached (5 requests/minute). "
                            "Please wait about 30-60 seconds and try again."
                        )

            # result['embedding'] can be a list of lists or a single list
            batch_emb = result["embedding"]
            if isinstance(batch_emb[0], list):
                embeddings.extend(batch_emb)
            else:
                embeddings.append(batch_emb)

        arr = np.array(embeddings, dtype=np.float32)
        # Normalize vectors for cosine similarity via Inner Product
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return arr / norms

    def _get_embedding_openai(self, texts: List[str]) -> np.ndarray:
        model = "text-embedding-3-small"
        response = self.openai_client.embeddings.create(input=texts, model=model)
        embeddings = [item.embedding for item in response.data]
        arr = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return arr / norms

    def get_embeddings(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        if not self.api_key:
            raise ValueError(f"API Key for {self.provider} is required to generate embeddings.")

        if self.provider == "gemini":
            task_type = "retrieval_query" if is_query else "retrieval_document"
            return self._get_embedding_gemini(texts, task_type=task_type)
        elif self.provider == "openai":
            return self._get_embedding_openai(texts)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def build_vector_index(self, chunks: List[DocumentChunk]):
        """
        Embeds chunks and creates a FAISS index (or TF-IDF index in Demo mode).
        """
        if not chunks:
            raise ValueError("No chunks provided to index.")

        self.chunks = chunks
        texts = [c.content for c in chunks]

        if self.provider in ["demo", "offline"]:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            return

        embeddings = self.get_embeddings(texts, is_query=False)
        self.embedding_dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        self.faiss_index.add(embeddings)

    def retrieve(self, query: str, top_k: int = 4) -> List[Tuple[DocumentChunk, float]]:
        """
        Retrieves top_k most similar chunks for the query along with cosine similarity score.
        """
        if self.provider in ["demo", "offline"]:
            if self.tfidf_matrix is None or not self.chunks:
                raise ValueError("Vector index is empty. Please upload and index a PDF first.")
            from sklearn.metrics.pairwise import cosine_similarity
            query_vec = self.tfidf_vectorizer.transform([query])
            sim_scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
            top_indices = np.argsort(sim_scores)[::-1][:min(top_k, len(self.chunks))]
            results = []
            for idx in top_indices:
                results.append((self.chunks[idx], float(sim_scores[idx])))
            return results

        if self.faiss_index is None or not self.chunks:
            raise ValueError("Vector index is empty. Please upload and index a PDF first.")

        query_vector = self.get_embeddings([query], is_query=True)
        top_k = min(top_k, len(self.chunks))
        scores, indices = self.faiss_index.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        return results

    def generate_answer(
        self,
        query: str,
        retrieved_items: List[Tuple[DocumentChunk, float]],
        temperature: float = 0.2,
    ) -> str:
        """
        Synthesizes an answer using the chosen LLM, grounded in the retrieved chunks.
        """
        if self.provider in ["demo", "offline"]:
            if not retrieved_items:
                return "The document does not contain enough matching text for this question."
            snippets = []
            for chunk, score in retrieved_items[:2]:
                first_few_sentences = ". ".join([s.strip() for s in chunk.content.split(". ") if s.strip()][:2])
                snippets.append(f"**From Page {chunk.page_number}** (Relevance: {score*100:.1f}%):\n> \"{first_few_sentences}\"")
            return "(Offline Demo Mode - Extracted Excerpts):\n\n" + "\n\n".join(snippets)

        if not self.api_key:
            raise ValueError("API Key is required to generate answers.")

        # Build context prompt
        context_parts = []
        for i, (chunk, score) in enumerate(retrieved_items, 1):
            context_parts.append(
                f"[Source Chunk #{i} | Page {chunk.page_number} | Relevance: {score:.3f}]\n{chunk.content}"
            )
        context_str = "\n\n---\n\n".join(context_parts)

        prompt = f"""You are a helpful and precise document assistant. Answer the user's question based strictly on the provided context excerpts from the uploaded PDF.
If the context does not contain enough information to answer the question, clearly state: "The uploaded document does not contain sufficient information to answer this question." Do not make assumptions or hallucinate.
Include citations to relevant page numbers where appropriate (e.g., [Page X]).

### Context Excerpts:
{context_str}

### Question:
{query}

### Answer:"""

        if self.provider == "gemini":
            import time
            from google.api_core import exceptions as google_exceptions
            import google.generativeai as genai
            
            model = genai.GenerativeModel("gemini-3.8-flash")
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = model.generate_content(
                        prompt,
                        generation_config={"temperature": temperature},
                    )
                    return response.text.strip()
                except google_exceptions.ResourceExhausted:
                    if attempt < max_retries - 1:
                        time.sleep(6 * (attempt + 1))
                    else:
                        raise RuntimeError(
                            "Gemini Free Tier rate limit reached (5 requests/minute). "
                            "Please wait a few seconds and submit your question again."
                        )

        elif self.provider == "openai":
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional RAG assistant who answers questions strictly based on provided PDF context.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()

        raise ValueError(f"Unsupported provider: {self.provider}")

    def query(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        """
        Full RAG pipeline: Retrieve + Generate.
        Returns a dict with 'question', 'answer', and 'sources'.
        """
        retrieved = self.retrieve(query, top_k=top_k)
        answer = self.generate_answer(query, retrieved)
        return {
            "question": query,
            "answer": answer,
            "sources": [
                {
                    "chunk_id": chunk.chunk_id,
                    "page": chunk.page_number,
                    "score": round(score, 4),
                    "content": chunk.content,
                }
                for chunk, score in retrieved
            ],
        }

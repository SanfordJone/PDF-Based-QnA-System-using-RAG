import re
import uuid

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "what",
    "when", "where", "which", "who", "why", "with",
}


def _terms(text):
    return {w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS}


class VectorStore:
    """
    Stores document chunks and finds the ones most relevant to a question.

    Uses ChromaDB embeddings (cosine similarity) when ChromaDB is installed,
    otherwise falls back to keyword overlap scoring.

    Each instance gets its own in-memory collection, so one user's uploads
    never appear in another user's search results and nothing is written to disk.
    """

    def __init__(self, use_chromadb=True):
        self.chunks = {}  # chunk_id -> {"text": ..., "metadata": ...}
        self.using_chromadb = False

        if use_chromadb:
            try:
                import chromadb

                self.client = chromadb.EphemeralClient()
                self.collection = self.client.get_or_create_collection(
                    name=f"pdf_{uuid.uuid4().hex}",
                    metadata={"hnsw:space": "cosine"},
                )
                self.using_chromadb = True
            except ImportError:
                pass

    def add_chunks(self, chunks, metadata=None):
        """
        Add a list of text chunks that belong to one document

        Args:
            chunks (list): Chunk strings
            metadata (dict or list): Shared metadata, or one dict per chunk

        Returns:
            list: IDs of the stored chunks
        """
        ids, texts, metas = [], [], []
        for i, text in enumerate(chunks):
            if not text or not text.strip():
                continue
            meta = dict(metadata[i] if isinstance(metadata, list) else (metadata or {}))
            chunk_id = str(uuid.uuid4())
            self.chunks[chunk_id] = {"text": text, "metadata": meta}
            ids.append(chunk_id)
            texts.append(text)
            metas.append(meta)

        if ids and self.using_chromadb:
            self.collection.add(ids=ids, documents=texts, metadatas=metas)
        return ids

    def search(self, query, k=4):
        """
        Find the chunks most relevant to the query

        Returns:
            list: Dicts with 'text', 'metadata', 'id' and 'score' (higher is better)
        """
        if not query or not query.strip() or not self.chunks:
            return []
        k = min(k, len(self.chunks))

        if self.using_chromadb:
            results = self.collection.query(query_texts=[query], n_results=k)
            matches = []
            for i, chunk_id in enumerate(results["ids"][0]):
                matches.append({
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] or {},
                    "id": chunk_id,
                    # cosine distance -> similarity
                    "score": 1 - results["distances"][0][i],
                })
            return matches

        # Keyword fallback: fraction of the question's terms found in each chunk
        query_terms = _terms(query)
        if not query_terms:
            return []
        scored = []
        for chunk_id, data in self.chunks.items():
            overlap = len(query_terms & _terms(data["text"]))
            if overlap:
                scored.append((overlap / len(query_terms), chunk_id))
        scored.sort(key=lambda s: s[0], reverse=True)
        return [
            {"text": self.chunks[cid]["text"], "metadata": self.chunks[cid]["metadata"], "id": cid, "score": score}
            for score, cid in scored[:k]
        ]

    def delete_document(self, filename):
        """Remove every chunk that came from the given file."""
        ids = [cid for cid, d in self.chunks.items() if d["metadata"].get("filename") == filename]
        for cid in ids:
            del self.chunks[cid]
        if ids and self.using_chromadb:
            self.collection.delete(ids=ids)
        return len(ids)

    def __len__(self):
        return len(self.chunks)

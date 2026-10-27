from vector_store import VectorStore


def make_store():
    store = VectorStore(use_chromadb=False)
    store.add_chunks(
        ["Refunds are processed within 14 business days.", "The head office is in Parramatta."],
        [{"filename": "policy.pdf", "page": 1}, {"filename": "policy.pdf", "page": 2}],
    )
    return store


def test_most_relevant_chunk_ranks_first():
    results = make_store().search("When are refunds processed?")
    assert results[0]["metadata"]["page"] == 1


def test_stopwords_alone_do_not_match():
    assert make_store().search("what is the") == []


def test_empty_query_and_empty_store():
    assert make_store().search("   ") == []
    assert VectorStore(use_chromadb=False).search("refunds") == []


def test_blank_chunks_are_skipped():
    store = VectorStore(use_chromadb=False)
    assert store.add_chunks(["", "   ", "real text"], {"filename": "a.pdf"}) and len(store) == 1


def test_delete_document_removes_its_chunks():
    store = make_store()
    assert store.delete_document("policy.pdf") == 2
    assert len(store) == 0


def test_sessions_do_not_share_documents():
    a = make_store()
    b = VectorStore(use_chromadb=False)
    assert len(a) == 2 and len(b) == 0
    assert b.search("refunds") == []

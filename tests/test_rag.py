from clairdoc_server.rag import chunk_text, cosine_similarity


def test_chunk_text_keeps_overlap() -> None:
    text = "A" * 600 + "\n" + "B" * 600
    chunks = chunk_text(text, size=500, overlap=100)

    assert len(chunks) >= 3
    assert chunks[0][-100:] == chunks[1][:100]


def test_cosine_similarity_ranks_same_direction_highest() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

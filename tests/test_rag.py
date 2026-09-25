from clairdoc_server.rag import chunk_text, cosine_similarity, extract_metadata, keyword_similarity


def test_chunk_text_keeps_overlap() -> None:
    text = "A" * 600 + "\n" + "B" * 600
    chunks = chunk_text(text, size=500, overlap=100)

    assert len(chunks) >= 3
    assert chunks[0][-100:] == chunks[1][:100]


def test_cosine_similarity_ranks_same_direction_highest() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_metadata_detects_invoice_date_and_amount() -> None:
    metadata = extract_metadata(
        "FACTURE ACME SARL\nDate : 14/03/2026\nTotal TTC 1 240,50 €",
        "facture.pdf",
    )

    assert metadata["category"] == "Factures"
    assert metadata["date"] == "2026-03-14"
    assert metadata["amounts"] == ["1 240,50 €"]


def test_keyword_similarity_ignores_accents() -> None:
    assert keyword_similarity("relevé bancaire", "RELEVE bancaire mensuel") == 1.0

from src.ingest import chunk_text, clean_text, process_file

from conftest import FIXTURES


def _words(n, prefix="w"):
    return " ".join(f"{prefix}{i}." for i in range(n))


def test_chunks_respect_size_limit():
    chunks = chunk_text(_words(1000), size=100, overlap=20, min_words=10)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 100 for c in chunks)


def test_no_data_loss():
    text = _words(733)
    chunks = chunk_text(text, size=120, overlap=30, min_words=10)
    covered = set(" ".join(chunks).split())
    assert covered == set(text.split())


def test_consecutive_chunks_overlap():
    chunks = chunk_text(_words(500), size=100, overlap=20, min_words=10)
    for a, b in zip(chunks, chunks[1:]):
        assert set(a.split()[-20:]) & set(b.split()[:20])


def test_overlap_must_be_smaller_than_size():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("hello world", size=10, overlap=10)


def test_short_document_kept():
    assert chunk_text("One short sentence about permits.", size=300, overlap=60) == [
        "One short sentence about permits."]


def test_clean_text_dehyphenates_and_removes_page_footers():
    out = clean_text("The plan-\nning scheme\nPage 3 of 10\napplies.")
    assert "planning" in out
    assert "Page 3" not in out


def test_html_boilerplate_removed_and_metadata_attached():
    meta = {"doc_id": "x", "title": "T", "url": "https://example.org", "activity_centre": "E"}
    chunks = process_file(FIXTURES / "exampleton_rezoning.html", meta)
    text = " ".join(c.text for c in chunks)
    assert "12 storeys" in text
    assert "Menu item" not in text and "Copyright footer" not in text
    assert chunks[0].url == "https://example.org"
    assert chunks[0].metadata()["page"] == -1


def test_ingest_reads_sources_csv(fixture_chunks):
    titles = {c.title for c in fixture_chunks}
    assert "Exampleton Fast-Track Permits" in titles
    assert all(c.url.startswith("https://example.org") for c in fixture_chunks)

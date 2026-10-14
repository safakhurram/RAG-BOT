from langchain_core.documents import Document

from src.sections import add_sections, search_text
from src.vectorstore import expand_query, tokenize


def page(n, text):
    return Document(page_content=text, metadata={"source": "p.pdf", "page": n})


def test_table_chunk_gets_programme_and_year_from_earlier_heading():
    pages = [
        page(0, "3.1.1 | Civil Engineering\nFIRST YEAR\nCE-112 | Engineering Drawing-I"),
        page(1, "THIRD YEAR\nCE-324 | Reinforced Concrete Design-I | 3 | 0 | 3"),
    ]
    chunk = Document(page_content=pages[1].page_content, metadata=dict(pages[1].metadata))

    add_sections(pages, [chunk])

    assert chunk.metadata["section"] == "3.1.1 Civil Engineering > THIRD YEAR"
    assert search_text(chunk).startswith("3.1.1 Civil Engineering > THIRD YEAR\n")
    assert "Civil" not in chunk.page_content  # quotable text is unchanged


def test_heading_number_on_its_own_line_and_toc_lines():
    pages = [page(0, "2.2.15 Department of Physics | 26\n3.1.2\nCivil Engineering (Specialisation in Construction)\nSECOND YEAR\nx")]
    chunk = Document(page_content="x", metadata=dict(pages[0].metadata))
    add_sections(pages, [chunk])
    assert chunk.metadata["section"] == "3.1.2 Civil Engineering (Specialisation in Construction) > SECOND YEAR"


def test_query_expansion_and_plurals():
    assert "third year" in expand_query("courses in 3 year of civil")
    assert "third year" in expand_query("3rd year courses")
    assert tokenize("Departments fees") == tokenize("department fee")

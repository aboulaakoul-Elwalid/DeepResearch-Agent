import chromadb
from chromadb.config import Settings

CHROMA_PATH = "/home/elwalid/projects/parallax_project/chroma_db"
COLLECTION = "arabic_books"
QUERY = "إسناد حديث في سنن أبي داود"
TOP_K = 3

client = chromadb.PersistentClient(
    path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False)
)
collection = client.get_collection(COLLECTION)
results = collection.query(
    query_texts=[QUERY],
    n_results=TOP_K,
    include=["documents", "metadatas", "distances"],
)
for doc, meta, dist in zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0],
):
    print("distance:", dist)
    print("book:", meta.get("book_title"), "pages:", meta.get("page_start"))
    print(doc[:400], "\n" + "-" * 80)

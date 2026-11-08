import hashlib
import json
from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader,
   
)
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": Docx2txtLoader,
    ".csv": CSVLoader,
}

def load_documents(data_dir="data", only_files=None):
    data_path = Path(data_dir)

    if not data_path.exists() or not data_path.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    all_documents = []

    for path in sorted(data_path.iterdir()):

        if not path.is_file() or path.suffix.lower() not in LOADERS:
            continue

        if only_files is not None and path.name not in only_files:
            continue

        try:
            all_documents.extend(load_document(path))
        except Exception as e:
            raise RuntimeError(
                f"Failed to load document: {path.name}. Error: {e}"
            ) from e

    return all_documents


def load_document(path):
    path = Path(path)

    extension = path.suffix.lower()

    if extension not in LOADERS:
        raise ValueError(f"Unsupported file type: {extension}")

    loader_class = LOADERS[extension]

    if extension == ".txt":
        loader = loader_class( str(path), autodetect_encoding=True,)
    else:
        loader = loader_class(str(path))

    documents = loader.load()

    for document in documents:
        document.metadata["source"] = path.name

    return documents



# ---------- Chunking ----------

def split_documents(documents, chunk_size=1000, chunk_overlap=150):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)

    return chunks

# ---------- Embeddings + vector store ----------
INDEX_DIR = "faiss_index"
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )

def get_vectorstore():
    index_path = Path(INDEX_DIR)

    faiss_file = index_path / "index.faiss"
    pickle_file = index_path / "index.pkl"

    if faiss_file.exists() and pickle_file.exists():
        return FAISS.load_local(str(index_path),get_embeddings(), allow_dangerous_deserialization=True,)

    return None
INDEXED_FILES_LOG = "indexed_files.json"


def calculate_file_hash(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for block in iter(lambda: file.read(4096), b""):
            sha256.update(block)

    return sha256.hexdigest()


def get_indexed_files():
    log_path = Path(INDEXED_FILES_LOG)

    if not log_path.exists():
        return {}

    indexed_files = json.loads(log_path.read_text())

    # Convert the old filename list into the new dictionary format
    if isinstance(indexed_files, list):
        migrated_files = {}

        for filename in indexed_files:
            file_path = Path("data") / filename

            if file_path.exists():
                migrated_files[filename] = calculate_file_hash(file_path)

        save_indexed_files(migrated_files)
        return migrated_files

    return indexed_files
    


def save_indexed_files(indexed_files):
    Path(INDEXED_FILES_LOG).write_text(
        json.dumps(indexed_files, indent=4)
    )


def mark_files_indexed(filenames):
    indexed = get_indexed_files()
    indexed.update(filenames)
    Path(INDEXED_FILES_LOG).write_text(json.dumps(list(indexed)))


def add_to_vectorstore(chunks, vectorstore=None):

    if vectorstore is None:
        vectorstore = get_vectorstore()

    if vectorstore is None:
        vectorstore = FAISS.from_documents(chunks,get_embeddings())
    else:
        vectorstore.add_documents(chunks)

    # Save the final updated index
    vectorstore.save_local(INDEX_DIR)

    return vectorstore

def delete_document_chunks(vectorstore, filename):
    ids_to_delete = [
        doc_id
        for doc_id, document in vectorstore.docstore._dict.items()
        if document.metadata.get("source") == filename
    ]

    if ids_to_delete:
        vectorstore.delete(ids=ids_to_delete)

    return len(ids_to_delete)
# ---------- Quick test ----------

if __name__ == "__main__":
    TEST_QUESTION = "What do adaptive approaches to digital ecosystems begin with?"  # replace with a real question from your data

    print("STEP 1: Loading documents...", flush=True)
    docs = load_documents()
    print(f"Loaded {len(docs)} documents", flush=True)

    print("STEP 2: Splitting documents...", flush=True)
    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunks", flush=True)

    print("STEP 3: Embedding and storing in FAISS...", flush=True)
    vectorstore = add_to_vectorstore(chunks)
    print("Stored and saved to disk", flush=True)

    print(f"STEP 4: Searching for: {TEST_QUESTION}", flush=True)
    for r in vectorstore.similarity_search(TEST_QUESTION, k=3):
        print(r.metadata, "->", r.page_content[:150], "\n", flush=True)
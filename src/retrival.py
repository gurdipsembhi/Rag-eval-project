import os
import glob
from pypdf import PdfReader

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()  # loads OPENAI_API_KEY from .env

DATA_DIR = "data/property_letters"
DB_DIR = "chroma_property_letters"


# 1. LOAD ---- extract text from each property-letter PDF
def load_property_letters():

    docs = []
    for path in sorted(glob.glob(f"{DATA_DIR}/*.pdf")):
        reader = PdfReader(path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        if not text:
            raise ValueError(f"No text found in {path}; scanned PDFs need OCR.")
        docs.append(Document(page_content=text, metadata={"source": os.path.basename(path)}))

    return docs


# 2. BUILD ---- chunk, embed once, and keep it on disk so we don't re-embed
def load_store():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    if os.path.exists(DB_DIR):
        return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    docs = load_property_letters()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    ).split_documents(docs)

    return Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)


def build_retriever():
    return load_store().as_retriever(search_kwargs={"k": 5})


# 3. TRY IT ---- python src/retriever.py
if __name__ == "__main__":

    retriever = build_retriever()

    results = retriever.invoke("What is asking price for (UAE-SYN-002)?")
    
    for r in results:
        print(f"[{r.metadata['source']}] {r.page_content[:150]}...\n")

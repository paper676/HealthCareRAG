from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

loader=PyPDFLoader('Abc.pdf')

docs=loader.load()

splitter=RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=60
)
chunks=splitter.split_documents(docs)

embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store =Chroma(
    embedding_function=embeddings,
    persist_directory='vectordb',
    collection_name='sample'
)
vector_store.add_documents(chunks)
vector_store.persist()


# print(len(chunks))
# text = chunks[0].page_content
# vector = embeddings.embed_query(text)

# print("Vector length:", len(vector))
# print("First 5 values:", vector)
# print(chunks[80].page_content)
print("Done")

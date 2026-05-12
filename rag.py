from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage
chat_history = []

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory='vectordb',
    collection_name='sample'
)

retriever = vector_store.as_retriever(search_kwargs={"k": 2})
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
def ask(q):
    
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        temperature=0.5,
        max_new_tokens=512
    )

    chat_model = ChatHuggingFace(llm=llm)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question using ONLY the context below. If you don't know, say 'I don't know',Do NOT infer or guess."),
        ("system", "Conversation history:\n{chat_history}"),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}")
    ])

    chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["query"])),
            "question": lambda x: x["query"],
            "chat_history": lambda _: "\n".join(
                [f"User: {m.content}" if isinstance(m, HumanMessage) else f"AI: {m.content}"
                for m in chat_history]
            )
        }
        | prompt
        | chat_model
    )

    result = chain.invoke({"query": q})
    chat_history.append(HumanMessage(content=q))
    chat_history.append(AIMessage(content=result.content))
    print("\nAnswer:\n", result.content)

if __name__ == "__main__":
    while True:
        q = input("Ask your Question (or type 'exit' to quit): ")
        if q.lower() == 'exit':
            break
        ask(q)
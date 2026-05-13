from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy
)
ragas_hf_llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="conversational",
    temperature=0.0,
    max_new_tokens=1024,
    timeout=120
)
ragas_llm = ChatHuggingFace(llm=ragas_hf_llm)

from langchain_core.messages import HumanMessage, AIMessage
chat_history = []

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory='vectordb',
    collection_name='sample'
)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def ask_question(q):
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        temperature=0.5,
        max_new_tokens=512
    )

    chat_model = ChatHuggingFace(llm=llm)

    rewrite=ChatPromptTemplate.from_messages([
        ("system","Rewrite the user promt to be better for document search."),
        ("human","{question}")
    ])
    rewriteChain= rewrite | chat_model
    rewriten_q=rewriteChain.invoke({"question": q}).content

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question using ONLY the context below,also use conversation history ONLY if not empty. If you don't know, say 'I don't know',Do NOT infer or guess."),
        # ("system", "Conversation history:\n{chat_history}"),
        ("human", "Conversation history:\n{chat_history}\n\nContext:\n{context}\n\nQuestion:\n{question}")
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
    
    docs=retriever.invoke(rewriten_q)

    result = chain.invoke({"query": rewriten_q})
    
    chat_history.append(HumanMessage(content=q))
    chat_history.append(AIMessage(content=result.content))

    ragas_dataset = Dataset.from_dict({
        "question": [rewriten_q],
        "answer": [result.content],
        "contexts": [[doc.page_content for doc in docs]]
    })

    scores = evaluate(
        ragas_dataset,
        metrics=[
            faithfulness,
            answer_relevancy
        ],
        llm=ragas_llm,
        embeddings=embeddings
    )
    evaluation = {
        k: float(v)
        for k, v in scores.scores[0].items()
    }
    return {
        "answer": result.content,
        "sources": [{"source": doc.metadata.get("source"),"page": doc.metadata.get("page")} for doc in docs],
        "evaluation": evaluation
    }
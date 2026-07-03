from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)


llm=ChatMistralAI(model="mistral-small-2506")

# Prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI assistant.

Use only the provided context to answer the question.

If the answer is not present in the context, say:
"I don't have enough information in the provided document."
"""
    ),
    (
        "human",
        """Context:
{context}

Question:
{question}
"""
    )
])

print("Rag system created")
print("Press 0 to exit")
while True:
    query = input("you : ")

    if query == "0":
        break

    docs = retriever.invoke(query)

    print(f"\nRetrieved Documents: {len(docs)}")

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    print("\nContext Preview:")
    print(context[:1000])

    print("\nGenerating answer...")

    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    response = llm.invoke(final_prompt)

    print("\nAI:", response.content)

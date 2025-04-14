import os
import pdfplumber
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema.runnable import RunnableConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load API key
load_dotenv()
CHATGROQ_API_KEY = os.getenv("groq_api_key")

# Load PDF using pdfplumber for better structure (table-aware)
def load_pdf_text(path):
    docs = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                docs.append(Document(page_content=text, metadata={"page": i + 1}))
    return docs

pages = load_pdf_text("New Joiner Connect SHI Onboarding.pdf")

# Split documents
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
docs = splitter.split_documents(pages)

# Embeddings & vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embedding=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Limit to top 3 chunks

# Define strict prompt with context
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an HR assistant. Use the following context to answer the user's question.\n"
        "If the answer is not explicitly in the context, say 'I’m not sure based on the provided information.'\n\n"
        "Context:\n{context}"
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# Format the retrieved chunks
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build input dict for chain
def build_inputs(x):
    docs = retriever.invoke(x["question"])
    print("\n--- Retrieved Context ---")
    for i, doc in enumerate(docs):
        print(f"[Doc {i+1} - Page {doc.metadata['page']}]:\n{doc.page_content[:300]}\n")
    return {
        "question": x["question"],
        "context": format_docs(docs),
        "chat_history": x["chat_history"]
    }

# LLM via Groq
llm = ChatGroq(
    groq_api_key=CHATGROQ_API_KEY,
    model="llama3-70b-8192",
    temperature=0.3
)

# Final chain: RAG + memory + output parser
rag_chain = (
    RunnableLambda(build_inputs)
    | prompt
    | llm
    | StrOutputParser()
)

# Store chat history
chat_histories = {}

def get_chat_history(session_id: str) -> ChatMessageHistory:
    if session_id not in chat_histories:
        chat_histories[session_id] = ChatMessageHistory()
    chat_histories[session_id].messages = chat_histories[session_id].messages[-7:]
    return chat_histories[session_id]

# Wrap with memory
rag_with_history = RunnableWithMessageHistory(
    rag_chain,
    get_chat_history,
    input_messages_key="question",
    history_messages_key="chat_history"
)

def answer_question(question, session_id="user1"):
    result=rag_with_history.invoke(
        {"question":question},
        config=RunnableConfig(configurable={"session_id":session_id})
    )
    return result

# Main chat loop
# print("🔹 HR Assistant Chatbot 🔹")
# while True:
#     question = input("You: ")
#     if question.lower() in {"exit", "quit"}:
#         break
#     result = rag_with_history.invoke(
#         {"question": question},
#         config=RunnableConfig(configurable={"session_id": "user1"})
#     )
#     print(f"Bot: {result}\n")

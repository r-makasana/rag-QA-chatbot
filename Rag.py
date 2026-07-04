from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()
# 1. Load PDF
loader = PyPDFLoader("E:\\Quantum Computing\\introduction-to-classical-and-quantum-computing-1e4p.pdf")
docs = loader.load()

# 2. Split
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. Embed + Store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()

# 4. LLM
llm = ChatGroq(model="llama-3.1-8b-instant")

# 5. Prompt
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.

Context: {context}

Question: {question}
""")

# 6. Chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 7. Ask
response = chain.invoke("What is this document about?")
print(response)
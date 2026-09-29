from langchain_chroma import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings
"""from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace"""
from langchain_core.messages import HumanMessage, SystemMessage
from huggingface_hub import login
from dotenv import load_dotenv


load_dotenv()

persistent_directory = 'db/class_db'

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", encode_kwargs={"normalize_embeddings": True})

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={'hnsw:space': 'cosine'}

)

query = input("How may I assist you: ")

retriever = db.as_retriever(search_kwargs={'k': 5})

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 5,
#         "score_threshold": 0.3  # Only return chunks with cosine similarity ≥ 0.3
#     }
# )


relevant_docs = retriever.invoke(query)

print(f'User Query: {query}')

print('---Context----')

for i, doc in enumerate(relevant_docs, start= 1):
    print(f'Document {i}: \n{doc.page_content}\n')

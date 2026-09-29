from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from huggingface_hub import login
from langchain_ollama import ChatOllama


load_dotenv()
persistent_directory = "db/class_db"

# Load embeddings and vector store
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", encode_kwargs={"normalize_embeddings": True})

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  
)

# Search for relevant documents
query = input('How may I assist you: ')

retriever = db.as_retriever(search_kwargs={"k": 5})

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 5,
#         "score_threshold": 0.3  # Only return chunks with cosine similarity ≥ 0.3
#     }
# )

relevant_docs = retriever.invoke(query)


# Display results
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")


# Combine the query and the relevant document contents
combined_input = f"""
The user's request is:

{query}


Based on the user's request generate the number of multiple-choice study questions asked based ONLY on
the information contained in the provided document chunks.

Requirements:
1. Use ONLY information found in the documents.
2. Do not invent facts or information.
3. Each question must have exactly 4 options.
4. Only ONE option must be correct.
5. Make the incorrect options plausible but clearly incorrect based
   on the provided documents.
6. Mix easy, medium, and difficult questions.
7. Cover different parts of the provided documents.
8. Do not repeat the same question.
9. The answer must be exactly one of the four options.
10. Return ONLY valid JSON.
11. Do NOT use Markdown code fences.
12. Do NOT include any explanation outside the JSON.

Return the result using exactly this structure:

{{
  "questions": [
    {{
      "id": 1,
      "question": "Question text",
      "options": [
        "A. Option 1",
        "B. Option 2",
        "C. Option 3",
        "D. Option 4"
      ],
      "answer": "Correct option letter (A, B, C, or D)"
    }}
  ]
}}

Document chunks:

{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}
"""
# Create a ChatOllama model

model = ChatOllama(
    model="qwen2.5:3b",
    temperature= 0
)


# Define the messages for the model
messages = [
    SystemMessage(content=
                  "You are a helpful teaching assistant, trained to assist the teacher in drafting questions for exams."),
    HumanMessage(content=combined_input),
]

# Invoke the model with the combined input
result = model.invoke(messages)

# Display the full result and content only
print("\n--- Generated Response ---")
# print("Full result:")
# print(result)
print("Content only:")
print(result.content)
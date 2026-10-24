
from langchain_openai import ChatOpenAI
import os

# Retrieve the top-k relevant chunks from the vector store
def retrieve(query: str, k: int = 2,vectorstore=None) -> list[str]:
    results = vectorstore.similarity_search(query, k=k)

    # Return each chunk with its content and metadata
    return [
        f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}"
        for r in results
    ]

# Define the generation function that prompts the LLM with retrieved context
def generate(query, retrieved_chunks,model) -> str:

    prompt = f"""
    You are an AI assistant answering questions about hotel policies.

    User Query:
    {query}

    Retrieved Policy Information:
    {retrieved_chunks}

    Rules:
    1. Use only the information in the retrieved policy information to answer the query. Do not rely on outside knowledge or assumptions.
    2. Base every statement on what is explicitly supported by the retrieved context, so the answer stays faithful to the source.
    3. Include all relevant conditions, exceptions, deadlines, fees, and limitations needed to answer the query completely.
    4. Stay focused on the user's specific question and do not add unrelated policy details.
    5. Write the answer in clear, concise, and professional language that hotel staff can relay directly to a guest.
    6. When the policy varies by rate plan, property, or situation, state the relevant condition rather than giving a single blanket answer.
    7. If the retrieved information is insufficient or does not cover the query, clearly say: \"The available policy documents do not contain enough information to answer this question.\"
    8. If the retrieved information is conflicting, point out the conflict instead of guessing which is correct.

    Answer:
    """

    return model.invoke(prompt)

# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 2,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    # Create the generator model
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE')
    )

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query=query, k=k,vectorstore=vectorstore)

    # Generate answer using retrieved chunks
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    return answer.content, retrieved_chunks

from langchain_google_vertexai import ChatVertexAI
from google.cloud import bigquery
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores import BigQueryVectorSearch
import os
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# Project and environment settings
PROJECT_ID = "solutions-data"
DATASET = "companyData"
TABLEEMBED = "order_em"
REGION = "asia-southeast1"

JSON_KEY_PATH = "credential/vertexAi.json"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = JSON_KEY_PATH


embedding_model = VertexAIEmbeddings(
    model_name="text-multilingual-embedding-002", project=PROJECT_ID
)
bq_vector_datasource = BigQueryVectorSearch(
    project_id=PROJECT_ID,
    dataset_name=DATASET,
    table_name=TABLEEMBED,
    location=REGION,
    embedding=embedding_model,
    content_field="text",
    text_embedding_field="embedding"
)

# Function to configure AI settings
def ai_config(model_name="gemini-1.5-pro-001", max_tokens=512, max_retries=6):
    llm = ChatVertexAI(model_name=model_name, max_tokens=max_tokens, max_retries=max_retries)


    config_prompt = """
        You are จิดริ้ด, a friendly AI assistant. 😊
        Context: {context}
        Input: {input}
        This is the context that you have discussed with the user before; use it as basic information to give the user an answer.


        User name: {user_name}
        You do not have to greet the user again.

        Using only provided information
        Do not mad up match data if it over your capability just inform user that
        Act like a human being and be able to talk with the user like normal. If you didn't ask about order lottery
        Counting lottery numbers Count the numbers according to the numbers sent out. Don't make a mistake.
        just keep conversation natural but short you here to assist with data order Lottery problem



        answer ih thai

        Refer to the user by their name.
    """

    return config_prompt, llm

# In-memory store for session history
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Function to batch retrieve results using fetch_k
def batch_retrieval_with_fetch_k(retriever, query, batch_size=100, fetch_k=200, num_batches=10):
    results = []
    for _ in range(num_batches):
        batch_results = retriever.get_relevant_documents(query, k=batch_size, fetch_k=fetch_k)
        results.extend(batch_results)
    return results

# Function to prompt AI and retrieve results
def prompt_ai(query, model_name="gemini-1.5-pro-001", session_id="", user_name="ลูกค้า", max_tokens=512, max_retries=100):

    context = ""  # Initialize an empty context
    first_time = True
    if session_id:
        message_history = get_session_history(session_id)
        if message_history and message_history.messages:
            context = " ".join([msg.content for msg in message_history.messages])
            first_time = False
        else:
            print("No message history found or message history is empty.")
            first_time = True
    else:
        print("No session ID provided, using default context.")

    # Debug context construction
    print(f"Constructed context: {context}")
    # print(f"First time: {first_time}")

    system_prompt, llm = ai_config(model_name, max_tokens, max_retries)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    retriever = bq_vector_datasource.as_retriever(search_type="mmr", search_kwargs={"k": 100, "fetch_k": 100})
    # retrieved_documents = batch_retrieval_with_fetch_k(retriever, query, batch_size=1000, fetch_k=1000, num_batches=10)

    # Convert the retrieved documents into a format expected by the question_answer_chain
    # documents = [doc.page_content for doc in retrieved_documents]

    history_aware_retriever = create_history_aware_retriever(llm, retriever, prompt)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # try:
    conversational_rag_chain = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    result = conversational_rag_chain.invoke(
        {"input": query, "context": context, "user_name": user_name},
        # {"input": query, "context": context, "user_name": user_name, "documents": documents},
        config={"configurable": {"session_id": session_id}},
    )
    answer = result.get("answer", "No answer found.")
    # except Exception as e:
    #     print(f"Error during invocation: {e}")
    #     answer = "An error occurred."

    # Debug result
    print(f"Result: {answer}")

    return answer

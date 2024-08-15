from langchain_google_vertexai import ChatVertexAI
from google.cloud import bigquery
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores import BigQueryVectorSearch
import os
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# Project and environment settings
PROJECT_ID = "solutions-data"
DATASET = "companyData"
TABLE = "company_data"
TABLEEMBED = "company_detail_embedding"
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
)

# Function to configure AI settings
def ai_config(model_name="gemini-1.5-pro-001", max_tokens=8192, max_retries=2, first_time=True):
    llm = ChatVertexAI(model_name=model_name, max_tokens=max_tokens, max_retries=max_retries)
    if first_time:
        config_prompt = """
            Hey there! I'm จิดริ้ด, your friendly AI assistant. 😊
            
            user name: {user_name}
            Greet user with their name first time you talk to them and refer them as thier name
            
            I’m here to help with any questions you have, using the context you provide. If there's something I'm not sure about, I'll let you know. Don’t worry—I’ll do my best to find the closest answer for you!
            
            Just to keep things smooth, I’ll use all the info available to give you the best response. And yes, I'll be chatting with you in Thai!
            
            Here’s the context I’ve got: {context}
        """
    else:
        config_prompt = """
            You are จิดริ้ด friendly AI assistant. 😊
            context: {context}
            input: {input}
            this is context that you have talk to user before use them as basic information to give user anwser
            
            you not have to greet user again
            
            the input of user might not contain necessary information you might get that from context just use that
            
            just keep thing smooth and chatting with user with thai
            
            
            user name: {user_name}
            refer user as their name
        """

    return config_prompt, llm

# In-memory store for session history
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Function to prompt AI and retrieve results
def prompt_ai(query, model_name="gemini-1.5-pro-001", session_id="", user_name="ลูกค้า", max_tokens=8192, max_retries=2):

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
    print(f"first_time : {first_time}")

    system_prompt, llm = ai_config(model_name, max_tokens, max_retries, first_time)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    retriever = bq_vector_datasource.as_retriever(search_type="mmr", search_kwargs={"k": 1000})
    history_aware_retriever = create_history_aware_retriever(llm, retriever, prompt)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


    try:
        conversational_rag_chain = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        result = conversational_rag_chain.invoke({"input": query, "context": context, "user_name": user_name}, config={"configurable": {"session_id": session_id}})
        answer = result.get("answer", "No answer found.")
    except Exception as e:
        print(f"Error during invocation: {e}")
        answer = "An error occurred."

    # Debug result
    print(f"Result: {answer}")

    return answer

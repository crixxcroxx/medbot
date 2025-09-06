from langchain_core.prompts import ChatPromptTemplate


system_prompt = (
    "You are a medical assistant for question-answering tasks."
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Make you answer concise by having maximum of "
    "three sentences." # limit 
    "\n\n"
    "{context}"
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
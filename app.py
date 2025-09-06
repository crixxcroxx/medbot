from flask import Flask, render_template, jsonify, request

from src import INDEX_NAME
from src.prompt import prompt_template
from src.helper import download_embedding, load_env_vars

from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


load_env_vars()
embedding = download_embedding()
docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embedding
)
retriever = docsearch.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 3}
)
chat_model = ChatGroq(model_name="groq/compound-mini", temperature=0)
qa_chain = create_stuff_documents_chain(chat_model, prompt_template)
rag_chain = create_retrieval_chain(retriever, qa_chain)


app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg")
    # return jsonify({"message": msg})
    
    response = rag_chain.invoke({"input": msg})
    return jsonify({"message": response["answer"]})
    

if __name__ == "__main__":
    app.run(host="localhost", port=8080, debug=True)


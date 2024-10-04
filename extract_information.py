from pymongo import MongoClient
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.document_loaders import DirectoryLoader
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import gradio as gr
from gradio.themes.base import Base
import os 
from dotenv import load_dotenv
load_dotenv()

client = MongoClient(os.environ['mango_uri'])
dbName = "langchain_demo"
collectionName = "collection_of_text_blobs"
collection = client[dbName][collectionName]

embeddings = OpenAIEmbeddings(openai_api_key = os.environ['openai_api_key'])

vectorStore = MongoDBAtlasVectorSearch(collection, embeddings)


def query_data(query):

    docs = vectorStore.similarity_search(query, K=1)
    as_ouput = docs[0].page_content

    llm = OpenAI(openai_api_key = os.environ['openai_api_key'], temperature = 0)
    retriever = vectorStore.as_retriever()
    qa = RetrievalQA.from_chain_type(llm , chain_type = "stuff", retriever = retriever)
    retriver_output = qa.run(query)

    return as_ouput, retriver_output

with gr.Blocks(theme = Base(), title='Question Answering App using Vector Search + RAG' ) as demo:
    gr.Markdown(
        """
        # Question Answering App using Vector Search + RAG
        """
    )
    textbox = gr.Textbox(label = "Enter your question here")
    with gr.Row():
        button = gr.Button("Submit", variant = "primary")
    with gr.Column():
        output1 = gr.Textbox(lines = 1, max_lines=10, label = "Answer from just Vector Search")
        output2 = gr.Textbox(lines = 1, max_lines=10, label = "Answer from RAG")

    button.click(query_data,textbox, outputs = [output1, output2])

demo.launch()




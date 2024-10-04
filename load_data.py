from pymongo import MongoClient
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.document_loaders import DirectoryLoader
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import gradio as gr
from gradio.themes.base import Base
from langchain.document_loaders import TextLoader
from dotenv import load_dotenv
load_dotenv()
import os

client = MongoClient(os.environ['mango_uri'])
dbName = "langchain_demo"
collectionName = "collection_of_text_blobs"
collection = client[dbName][collectionName]

loader = TextLoader('./sample_files/aerodynamics.txt')
data = loader.load()

embeddings = OpenAIEmbeddings(openai_api_key =os.environ['openai_api_key'])

vectorStore = MongoDBAtlasVectorSearch.from_documents(data,embeddings,collection = collection)
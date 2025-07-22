from dotenv import load_dotenv
import os
load_dotenv()

# Modern import
from langchain_openai import OpenAIEmbeddings

def get_embedder():
    openai_key = os.getenv("OPENAI_API_KEY")
    return OpenAIEmbeddings(openai_api_key=openai_key)

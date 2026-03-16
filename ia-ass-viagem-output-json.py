from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("API_KEY")

prompt_cidade = ChatPromptTemplate.from_template("""
    Sugira uma cidade dado o meu interesse por {interesse}
""")

llm = ChatOllama(
    model="llama3:8b",
    temperature=0.3,
    api_key=api_key
)

chain = prompt_cidade | llm | StrOutputParser()

resposta = chain.invoke(
    {
        "interesse": "praias"
    }
)

print(resposta)
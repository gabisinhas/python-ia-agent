from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

app = FastAPI()

llm = ChatOllama(
    model="llama3:8b",
    temperature=0.3
)

class Prompt(BaseModel):
    message: str

@app.post("/chat")
def chat(prompt: Prompt):
    messages = [
        SystemMessage(content="Responda de forma clara e objetiva em português."),
        HumanMessage(content=prompt.message)
    ]

    response = llm.invoke(messages)

    return {
        "response": response.content
    }

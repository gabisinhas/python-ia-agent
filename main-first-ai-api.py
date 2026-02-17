from fastapi import FastAPI
from pydantic import BaseModel
import ollama

app = FastAPI()

class Prompt(BaseModel):
    message: str

@app.post("/chat")
def chat(prompt: Prompt):
    response = ollama.chat(
        model="llama3:8b",
        messages=[
            {"role": "system", "content": "Responda de forma objetiva e em português."},
            {"role": "user", "content": prompt.message}
        ]
    )
    print(response["message"]["content"])
    return {"response": response["message"]["content"]}

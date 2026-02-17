from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# ================================
# 1. BASE DE CONHECIMENTO (TXT)
# ================================

loader = TextLoader("assistente_viagem_rag_estudo.txt", encoding="utf-8")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="llama3")

vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ================================
# 2. PROMPT
# ================================

prompt = ChatPromptTemplate.from_template("""
Você é um assistente de IA especializado em planejamento de viagens.

Use APENAS as informações abaixo para responder.

Contexto:
{context}

Pergunta do usuário:
{question}

Se a resposta não estiver no contexto, diga que não encontrou a informação.
""")

# ================================
# 3. LLM
# ================================

llm = ChatOllama(
    model="llama3:8b",
    temperature=0.3
)

# ================================
# 4. LOOP RAG
# ================================

def iniciar_assistente_rag():
    print("Assistente de Viagem com RAG (Estudo)")
    print("Digite 'sair' para encerrar.\n")

    while True:
        pergunta = input("Você: ")

        if pergunta.lower() in ["sair", "exit"]:
            print("Assistente: Até mais! ✈️")
            break

        # Recuperação
        docs = retriever.invoke(pergunta)
        contexto = "\n".join(d.page_content for d in docs)

        # Prompt final
        mensagens = prompt.format_messages(
            context=contexto,
            question=pergunta
        )

        resposta = llm.invoke(mensagens)

        print("\nAssistente:", resposta.content)
        print("-" * 50)


if __name__ == "__main__":
    iniciar_assistente_rag()

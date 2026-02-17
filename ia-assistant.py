from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

template = """Você é um assistente de IA que ajuda o usuário a planejar uma viagem, dando sugestões de destinos, roteiros
e dicas práticas. 

Siga este fluxo de conversa:
1. Primeiro, pergunte ao usuário qual o país e cidade de destino da viagem
2. Depois, pergunte quantos dias ele pretende ficar
3. Então, crie um roteiro personalizado baseado no destino e duração informados, incluindo dicas de transporte, alimentação e hospedagem

Seja cordial e organize bem as informações para facilitar o planejamento do usuário.

Histórico da conversa: {history}

Entrada do usuário: {input}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ]
)

llm = ChatOllama(
    model="llama3:8b",
    temperature=0.3
)

chain = prompt | llm

## Armazena o histórico da conversa
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)


def iniciar_assistente_viagem():
    print("Bem-vindo ao Assistente de Viagem! Digite 'sair' para encerrar.\n")

    # Inicia a conversa com uma mensagem vazia para o assistente começar
    resposta_inicial = chain_with_history.invoke(
        {'input': 'Olá'},
        config={'configurable': {'session_id': 'user123'}}
    )
    print('Assistente de Viagem:', resposta_inicial.content)
    print()

    while True:
        pergunta_usuario = input("Você: ")
        if pergunta_usuario.lower() in ["sair", "exit"]:
            print("Assistente de Viagem: Até mais! Aproveite sua viagem!")
            break

        resposta = chain_with_history.invoke(
            {'input': pergunta_usuario},
            config={'configurable': {'session_id': 'user123'}}
        )

        print('Assistente de Viagem:', resposta.content)
        print()


if __name__ == '__main__':
    iniciar_assistente_viagem()
# IMPORTAÇÃO DAS BIBLIOTECAS NECESSÁRIAS
import os
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory                   # Permite criar Históricos de mensagens
from langchain_core.chat_history import BaseChatMessageHistory                              # Classe base para histórico de mensagens
from langchain_core.runnables.history import RunnableWithMessageHistory                     # Permite gerenciar o histórico de mensagens
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder                  # Permite criar prompts / mensagens
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages   # Mensagens humanas, do sistema e do AI
from langchain_core.runnables import RunnablePassthrough                                    # Permite criar fluxos de execução e reutilizaveis
from operator import itemgetter                                                             # Facilita a extração de valores de dicionários


# Carregar as variáveis de ambiente do arquvo .env (para proteger as credenciais)
load_dotenv(find_dotenv())

# Obter a chave da API do GROQ armazenada no arquivo .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Inicializar o modelo de AI utilizando a API da GROQ
model = ChatGroq(
    model = "gemma2-9b-it",
    groq_api_key = GROQ_API_KEY
)

# Dicionário para armazenar o histórico de mensagens
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Recura ou cria um histórico de mansagens para uma determinada sesão.
    Isso permite manter o contexto contínuo para diferentes usuários e interações.
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Criar um gerenciador de histórico que conecta o modelo ao armazenamento de mensagens
with_message_history = RunnableWithMessageHistory(model, get_session_history)

# Configuração da sessão (Identificador único para cada chat/usuário)
config = {"configurable":{"session_id":"chat1"}}

# Exemplo de interação inicial do usuário
response = with_message_history.invoke(
    [HumanMessage(content="Oi, meu nome é Alan e sou cientista de dados.")],
    config=config
)

# Exibir a resposta do modelo
print("Resposta do modelo:", response)

# Criação de um prompt template para estruturar a entrada do modelo
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um assistente útil. Responda todas as perguntas com precisão."),
        MessagesPlaceholder(variable_name="messages")  # Permite adicionar mensagens dinamicamente
    ]
)

# Conecta o modelo ao template de prompt
chain = prompt | model

# Exemplo de interação usando o template
chain.invoke({"messages": [HumanMessage(content="Oi, meu nome é Alan")]})

# Gerenciamento da memória do chatbot
trimmer = trim_messages(
    max_tokens=45,        # Define um limite máximo de tokens para evitar ultrapassar o consumo de memória
    strategy="last",      # Mantém só as últimas mensagens mais recentes
    token_counter=model,  # Usa o modelo para contar os tokens
    include_system=True,  # Inclui a mensagem do sistema no histórico
    allow_partial=False,  # Evita que mensagens fiquem cortadas
    start_on="human"      # Começa a contagem com mensagens humanas
)

# Exemplo de histórico de mensagens
messages = [
    SystemMessage(content="Você é um bom assistente"),
    HumanMessage(content="Oi! Meu nome é Bob"),
    AIMessage(content="Oi, Bob! Como posso te ajudar hoje?"),
    HumanMessage(content="Eu gosto de sorvete de baunilha"),
]

# Aplica o limitador de memória ao histórico de mensagens
trimmer.invoke(messages)

# Criando um pipeline de execução para otimizar a passagem de informações
chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)  # Aplica a otimização do histórico
    | prompt  # Passa a entrada pelo template de prompt
    | model  # Envia para o modelo
)

# Exemplo de interação utilizando o pipeline otimizado
response = chain.invoke(
    {
        "messages": messages + [HumanMessage(content="Qual sorvete eu gosto?")],
        "language": "Português"
    }
)

# Exibe a resposta final do modelo
print("Resposta final:", response.content)




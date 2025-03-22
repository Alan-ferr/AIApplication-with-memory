# README - Chatbot com Groq e LangChain

Este projeto implementa um chatbot inteligente utilizando a API da Groq e componentes do LangChain para gerenciamento de contexto e memória. O sistema permite interações conversacionais contínuas com histórico persistente e otimização de recursos.

## Funcionalidades Principais
- **Integração com Groq**: Utilização do modelo `gemma2-9b-it` para geração de respostas
- **Gerenciamento de Histórico**: Manutenção de contexto por sessão de conversa
- **Otimização de Memória**: Controle de consumo de tokens com estratégia de truncagem
- **Templates Estruturados**: Criação de prompts dinâmicos com instruções do sistema
- **Pipeline Modular**: Fluxo de processamento configurável com componentes reutilizáveis

## Pré-requisitos
- Python 3.8+
- Conta na [Groq Cloud](https://console.groq.com/)
- API Key da Groq

## Instalação
```bash
pip install langchain-groq python-dotenv langchain-core
```

## Configuração
1. Crie um arquivo `.env` na raiz do projeto:
```env
GROQ_API_KEY=sua_chave_aqui
```

## Explicação Detalhada do Código

### 1. Importação de Bibliotecas
```python
import os
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
```
- **Bibliotecas Principais**: Gerenciamento de ambiente, integração Groq e componentes LangChain
- **Tipos de Mensagem**: `HumanMessage` (usuário), `AIMessage` (assistente), `SystemMessage` (instruções)
- **Gerenciadores de Fluxo**: `RunnablePassthrough` e `RunnableWithMessageHistory` para controle de execução

### 2. Configuração Inicial
```python
load_dotenv(find_dotenv())
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    model="gemma2-9b-it",
    groq_api_key=GROQ_API_KEY
)
```
- Carrega variáveis de ambiente do arquivo `.env`
- Inicializa o modelo Groq com a chave API

### 3. Gerenciamento de Histórico (Exemplo 1)
```python
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

with_message_history = RunnableWithMessageHistory(model, get_session_history)
```
- **Armazenamento em Memória**: Dicionário para manter históricos por sessão
- **Criação de Contexto**: Sistema mantém conversas separadas usando `session_id`
- **Configuração de Sessão**: Exemplo usando `session_id="chat1"`

### 4. Templates e Memória (Exemplo 2)
```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente útil..."),
    MessagesPlaceholder(variable_name="messages")
])

trimmer = trim_messages(
    max_tokens=45,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human"
)

chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt
    | model
)
```
- **Template Dinâmico**: Combina instruções fixas com histórico de conversa
- **Otimização de Memória**: Mantém apenas últimas mensagens dentro do limite de tokens
- **Pipeline de Execução**: Fluxo completo desde entrada até resposta final

## Exemplos de Uso

### Exemplo 1: Conversa Básica
```python
response = with_message_history.invoke(
    [HumanMessage(content="Oi, sou Alan e cientista de dados.")],
    config={"configurable":{"session_id":"chat1"}}
)
print("Resposta:", response.content)
```

### Exemplo 2: Conversa Contextual
```python
messages = [
    SystemMessage(content="Você é um assistente de comida."),
    HumanMessage(content="Gosto de sorvete de baunilha"),
]

response = chain.invoke({
    "messages": messages + [HumanMessage(content="Qual sorvete eu gosto?")]
})
print("Resposta:", response.content)
```

## Notas Adicionais
1. **Gestão de Tokens**: Ajuste `max_tokens` conforme necessidades de desempenho
2. **Persistência de Dados**: O histórico atual é armazenado em memória (reinicia com o programa)
3. **Segurança**: Mantenha a API Key protegida no arquivo `.env`
4. **Modelos Suportados**: Experimente outros modelos da Groq como `mixtral-8x7b-32768`

Este projeto oferece uma base flexível para desenvolvimento de chatbots inteligentes com controle de contexto e otimização de recursos. Adapte os templates e parâmetros conforme suas necessidades específicas.
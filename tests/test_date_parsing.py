from datetime import datetime, timedelta
import os
import sys
import types

# Crear módulos ficticios para evitar dependencias pesadas al importar
dummy_names = [
    "supabase",
    "langchain_openai",
    "langchain_community",
    "langchain_community.vectorstores",
    "langchain.chains",
    "langchain.memory",
    "langchain.prompts",
    "langchain.schema",
    "langchain.callbacks",
    "langchain.text_splitter",
    "pydantic",
    "dotenv",
]

for name in dummy_names:
    if name not in sys.modules:
        module = types.ModuleType(name)
        sys.modules[name] = module

sys.modules["supabase"].create_client = lambda *a, **k: None
sys.modules["langchain.schema"].Document = type("Document", (), {})
sys.modules["langchain.schema"].BaseRetriever = type("BaseRetriever", (), {})
sys.modules["langchain_community.vectorstores"].SupabaseVectorStore = object
sys.modules["langchain.chains"].ConversationalRetrievalChain = object
sys.modules["langchain.memory"].ConversationBufferMemory = object
sys.modules["langchain.prompts"].PromptTemplate = object
sys.modules["langchain.callbacks"].StreamingStdOutCallbackHandler = object
sys.modules["langchain.text_splitter"].RecursiveCharacterTextSplitter = object
sys.modules["pydantic"].BaseModel = object
sys.modules["dotenv"].load_dotenv = lambda *a, **k: None
sys.modules["langchain_openai"].ChatOpenAI = lambda *a, **k: None
sys.modules["langchain_openai"].OpenAIEmbeddings = lambda *a, **k: None

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from chatbot import QueryProcessor, parse_date_command


def test_parse_date_command_del_al():
    assert parse_date_command("del 1-5-2024 al 3-6-2024") == (
        "2024-05-01",
        "2024-06-03",
    )


def test_parse_date_command_dots():
    assert parse_date_command("desde 01.05.2024 hasta 03.06.2024") == (
        "2024-05-01",
        "2024-06-03",
    )


def test_extract_single_iso_date():
    result = QueryProcessor.extract_date_filters("alertas de 2024-05-01")
    assert result == {"specific_date": "2024-05-01"}


def test_extract_relative_without_accents():
    today = datetime.now()
    expected_from = (today - timedelta(days=3)).strftime("%Y-%m-%d")
    expected_to = today.strftime("%Y-%m-%d")
    result = QueryProcessor.extract_date_filters("ultimos 3 dias")
    assert result == {"date_from": expected_from, "date_to": expected_to}


def test_extract_month_abbreviation():
    result = QueryProcessor.extract_date_filters("alertas de mar 2024")
    assert result["date_from"].startswith("2024-03") and result["date_to"].startswith("2024-03")

import importlib
import sys
from types import ModuleType, SimpleNamespace
import pytest
from unittest import mock


def setup_stub_modules(monkeypatch):
    import os
    root = os.path.dirname(os.path.dirname(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)
    # Create simple stub modules for missing dependencies
    modules = {
        'supabase': ModuleType('supabase'),
        'langchain_openai': ModuleType('langchain_openai'),
        'langchain_community': ModuleType('langchain_community'),
        'langchain_community.vectorstores': ModuleType('langchain_community.vectorstores'),
        'langchain.chains': ModuleType('langchain.chains'),
        'langchain.memory': ModuleType('langchain.memory'),
        'langchain.prompts': ModuleType('langchain.prompts'),
        'langchain.schema': ModuleType('langchain.schema'),
        'langchain.callbacks': ModuleType('langchain.callbacks'),
        'langchain.text_splitter': ModuleType('langchain.text_splitter'),
        'pydantic': ModuleType('pydantic'),
        'dotenv': ModuleType('dotenv'),
    }
    for name, module in modules.items():
        sys.modules[name] = module

    # Minimal attributes for imports
    sys.modules['supabase'].create_client = lambda *a, **k: mock.MagicMock()
    class Dummy:
        def __init__(self, *a, **k):
            pass
    sys.modules['langchain_openai'].ChatOpenAI = Dummy
    sys.modules['langchain_openai'].OpenAIEmbeddings = Dummy
    sys.modules['langchain_community'].vectorstores = sys.modules['langchain_community.vectorstores']
    sys.modules['langchain_community.vectorstores'].SupabaseVectorStore = Dummy
    sys.modules['langchain.chains'].ConversationalRetrievalChain = type('CRC', (), {'from_llm': classmethod(lambda cls, **k: mock.MagicMock())})
    sys.modules['langchain.memory'].ConversationBufferMemory = Dummy
    sys.modules['langchain.prompts'].PromptTemplate = Dummy
    sys.modules['langchain.callbacks'].StreamingStdOutCallbackHandler = Dummy
    sys.modules['langchain.text_splitter'].RecursiveCharacterTextSplitter = Dummy
    # Provide minimal attributes used in chatbot
    sys.modules['dotenv'].load_dotenv = lambda *a, **k: None
    sys.modules['pydantic'].BaseModel = object
    sys.modules['langchain.schema'].Document = type('Document', (), {'__init__': lambda self, page_content, metadata: setattr(self, 'page_content', page_content) or setattr(self, 'metadata', metadata)})
    sys.modules['langchain.schema'].BaseRetriever = object


def import_chatbot(monkeypatch):
    setup_stub_modules(monkeypatch)
    chatbot = importlib.import_module('chatbot')
    return chatbot


@pytest.fixture
def chatbot_module(monkeypatch):
    # Ensure a fresh import for each test
    if 'chatbot' in sys.modules:
        del sys.modules['chatbot']
    return import_chatbot(monkeypatch)


def make_query_mock():
    q = mock.MagicMock()
    q.select.return_value = q
    q.filter.return_value = q
    q.gte.return_value = q
    q.lte.return_value = q
    q.eq.return_value = q
    q.order.return_value = q
    q.limit.return_value = q
    return q


def test_search_by_date_range_builds_query(monkeypatch, chatbot_module):
    cb = chatbot_module
    query_mock = make_query_mock()
    query_mock.execute.return_value = SimpleNamespace(data=[{
        'id': 1,
        'title': 't',
        'description': 'd',
        'presentation_date': '2024-01-01',
        'created_at': 'c',
        'category': 'cat',
        'country': 'AR',
        'institution': 'inst'
    }])

    supabase_mock = mock.MagicMock()
    supabase_mock.from_.return_value = query_mock
    monkeypatch.setattr(cb, 'supabase', supabase_mock)

    results = cb.ImprovedChatbot.search_by_date_range(cb.ImprovedChatbot, '2024-01-01', '2024-01-31', 'AR')

    supabase_mock.from_.assert_called_once_with('alertas')
    query_mock.select.assert_called_once()
    query_mock.filter.assert_any_call('presentation_date', 'gte', '2024-01-01')
    query_mock.filter.assert_any_call('presentation_date', 'lte', '2024-01-31')
    query_mock.eq.assert_called_once_with('country', 'AR')
    query_mock.order.assert_called_once_with('presentation_date', desc=True)
    query_mock.execute.assert_called_once()
    assert results[0]['id'] == 1


def test_search_by_date_range_handles_exception(monkeypatch, chatbot_module):
    cb = chatbot_module
    query_mock = make_query_mock()
    query_mock.execute.side_effect = Exception('fail')
    supabase_mock = mock.MagicMock()
    supabase_mock.from_.return_value = query_mock
    monkeypatch.setattr(cb, 'supabase', supabase_mock)

    results = cb.ImprovedChatbot.search_by_date_range(cb.ImprovedChatbot, '2024-01-01', '2024-01-31')
    assert results == []


def test_test_connection_success(monkeypatch, chatbot_module):
    cb = chatbot_module
    query_mock = make_query_mock()
    query_mock.execute.return_value = SimpleNamespace(count=5)
    supabase_mock = mock.MagicMock()
    supabase_mock.from_.return_value = query_mock
    monkeypatch.setattr(cb, 'supabase', supabase_mock)

    assert cb.test_connection() is True
    supabase_mock.from_.assert_called_once_with('alertas')
    query_mock.select.assert_called_once_with('count', count='exact')
    query_mock.execute.assert_called_once()


def test_test_connection_failure(monkeypatch, chatbot_module):
    cb = chatbot_module
    query_mock = make_query_mock()
    query_mock.execute.side_effect = Exception('db error')
    supabase_mock = mock.MagicMock()
    supabase_mock.from_.return_value = query_mock
    monkeypatch.setattr(cb, 'supabase', supabase_mock)

    assert cb.test_connection() is False


def test__search_by_date_builds_query(monkeypatch, chatbot_module):
    cb = chatbot_module
    query_mock = make_query_mock()
    query_mock.execute.return_value = SimpleNamespace(data=[{
        'id': 1,
        'title': 't',
        'description': 'd',
        'presentation_date': '2024-01-01',
        'created_at': 'c',
        'category': 'cat',
        'country': 'AR',
        'institution': 'inst'
    }])
    supabase_mock = mock.MagicMock()
    supabase_mock.from_.return_value = query_mock
    monkeypatch.setattr(cb, 'supabase', supabase_mock)

    vectorstore = mock.MagicMock()
    retriever = cb.CustomRetrieverWithDateFilter()
    retriever.vectorstore = vectorstore
    retriever.k = 5
    docs = retriever._search_by_date('hola', {'specific_date': '2024-01-01'})

    supabase_mock.from_.assert_called_with('alertas')
    query_mock.eq.assert_called_with('presentation_date', '2024-01-01')
    query_mock.execute.assert_called_once()
    assert len(docs) == 1
    assert docs[0].metadata['id'] == 1


def test__search_by_date_exception_fallback(monkeypatch, chatbot_module):
    cb = chatbot_module
    query_mock = make_query_mock()
    query_mock.execute.side_effect = Exception('oops')
    supabase_mock = mock.MagicMock()
    supabase_mock.from_.return_value = query_mock
    monkeypatch.setattr(cb, 'supabase', supabase_mock)

    vectorstore = mock.MagicMock()
    vectorstore.similarity_search.return_value = ['fallback']
    retriever = cb.CustomRetrieverWithDateFilter()
    retriever.vectorstore = vectorstore
    retriever.k = 5
    docs = retriever._search_by_date('hola', {'specific_date': '2024-01-01'})

    assert docs == ['fallback']
    vectorstore.similarity_search.assert_called_once()


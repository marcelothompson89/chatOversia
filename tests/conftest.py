import sys
import types

class Dummy(types.ModuleType):
    def __getattr__(self, name):
        return type(name, (), {"__init__": lambda self, *a, **k: None})

    def __call__(self, *args, **kwargs):
        return None

# List of modules to fake
modules = [
    'supabase',
    'langchain_openai',
    'langchain_community',
    'langchain_community.vectorstores',
    'langchain.chains',
    'langchain.memory',
    'langchain.prompts',
    'langchain.schema',
    'langchain.callbacks',
    'langchain.text_splitter',
    'dotenv',
    'pydantic'
]

for mod in modules:
    parts = mod.split('.')
    for i in range(1, len(parts)+1):
        sub = '.'.join(parts[:i])
        if sub not in sys.modules:
            sys.modules[sub] = Dummy(sub)

# provide a no-op load_dotenv
sys.modules['dotenv'].load_dotenv = lambda *a, **k: None

# specific helpers
sys.modules['supabase'].create_client = lambda *a, **k: object()

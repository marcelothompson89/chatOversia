import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
import re
from supabase import create_client
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document, BaseRetriever
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from dotenv import load_dotenv

# ─── CONFIGURACIÓN ───────────────────────────────────────────────────────────────
# Cargar variables de entorno
load_dotenv(".env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

required_vars = {
    "SUPABASE_URL": SUPABASE_URL,
    "SUPABASE_KEY": SUPABASE_KEY,
    "OPENAI_API_KEY": OPENAI_API_KEY,
}
missing = [name for name, val in required_vars.items() if not val]
if missing:
    missing_str = ", ".join(missing)
    raise SystemExit(
        f"Missing environment variables: {missing_str}."
        " Please set them in your .env file or environment."
    )

# ─── CLIENTES ────────────────────────────────────────────────────────────────────
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ─── PROMPT PERSONALIZADO ────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Eres un asistente experto en alertas regulatorias. Tu función es ayudar a los usuarios a:
- Buscar alertas específicas por palabras clave, fechas o temas
- Crear resúmenes ejecutivos de múltiples alertas
- Identificar patrones y tendencias regulatorias
- Explicar el impacto y contexto de las regulaciones

IMPORTANTE: El contexto a continuación contiene las alertas encontradas para la consulta del usuario. DEBES usar esta información para responder.

Reglas estrictas:
1. SI HAY INFORMACIÓN en el contexto, DEBES presentarla como alertas encontradas para la fecha/criterio solicitado
2. Cada alerta en el contexto corresponde a la búsqueda realizada
3. NUNCA digas "no se encontraron alertas" si hay contenido en el contexto
4. Presenta TODAS las alertas del contexto de manera organizada
5. Las alertas mostradas abajo SON del período solicitado por el usuario

Contexto con las alertas encontradas:
{context}

Historial: {chat_history}

Pregunta: {question}

Responde presentando las alertas encontradas en el contexto de manera clara y detallada:"""

# ─── PROCESADOR DE CONSULTAS ─────────────────────────────────────────────────────
class QueryProcessor:
    """Procesa las consultas del usuario para extraer intenciones y filtros"""
    
    @staticmethod
    def extract_date_filters(query: str) -> Optional[Dict[str, Any]]:
        """Extrae filtros de fecha de la consulta"""
        filters = {}
        query_lower = query.lower()
        
        # Detectar rangos explícitos "del X al Y" o "desde X hasta Y"
        range_pattern = r'(?:del|desde)\s+(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{4})\s+(?:al|hasta)\s+(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{4})'
        match = re.search(range_pattern, query_lower)
        if match:
            date1, date2 = match.groups()
            filters["date_from"] = normalize_date(date1)
            filters["date_to"] = normalize_date(date2)
            return filters

        # Detectar fechas específicas en formatos DD/MM/YYYY, DD-MM-YYYY,
        # DD.MM.YYYY o YYYY-MM-DD
        date_patterns = [
            r'(\d{1,2})[\/\.-](\d{1,2})[\/\.-](\d{4})',  # DD/MM/YYYY o DD-MM-YYYY
            r'(\d{4})[\/\.-](\d{1,2})[\/\.-](\d{1,2})',  # YYYY-MM-DD o YYYY/MM/DD
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, query)
            if matches:
                if len(matches) == 1:
                    # Una sola fecha
                    if pattern == date_patterns[0]:  # DD/MM/YYYY
                        day, month, year = matches[0]
                    else:  # YYYY-MM-DD
                        year, month, day = matches[0]
                    filters["specific_date"] = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        
        # Detectar períodos relativos
        if not filters:
            today = datetime.now()
            if "último mes" in query_lower or "ultimo mes" in query_lower or "mes pasado" in query_lower:
                filters["date_from"] = (today - timedelta(days=30)).strftime("%Y-%m-%d")
                filters["date_to"] = today.strftime("%Y-%m-%d")
            elif "última semana" in query_lower or "ultima semana" in query_lower or "semana pasada" in query_lower:
                filters["date_from"] = (today - timedelta(days=7)).strftime("%Y-%m-%d")
                filters["date_to"] = today.strftime("%Y-%m-%d")
            elif ("últimos" in query_lower or "ultimos" in query_lower) and ("días" in query_lower or "dias" in query_lower):
                # Buscar número de días
                match = re.search(r'(?:últimos|ultimos)\s+(\d+)\s+(?:días|dias)', query_lower)
                if match:
                    days = int(match.group(1))
                    filters["date_from"] = (today - timedelta(days=days)).strftime("%Y-%m-%d")
                    filters["date_to"] = today.strftime("%Y-%m-%d")
            elif any(re.search(rf"\b{m}\b", query_lower) for m in [
                "enero", "ene", "febrero", "feb", "marzo", "mar", "abril", "abr", "mayo", "may", "junio", "jun",
                "julio", "jul", "agosto", "ago", "septiembre", "sep", "set", "octubre", "oct", "noviembre", "nov",
                "diciembre", "dic"]):
                # Detectar mes específico
                months_map = {
                    "enero": 1, "ene": 1,
                    "febrero": 2, "feb": 2,
                    "marzo": 3, "mar": 3,
                    "abril": 4, "abr": 4,
                    "mayo": 5, "may": 5,
                    "junio": 6, "jun": 6,
                    "julio": 7, "jul": 7,
                    "agosto": 8, "ago": 8,
                    "septiembre": 9, "sep": 9, "set": 9,
                    "octubre": 10, "oct": 10,
                    "noviembre": 11, "nov": 11,
                    "diciembre": 12, "dic": 12
                }
                for month_name, month_num in months_map.items():
                    if re.search(rf"\b{month_name}\b", query_lower):
                        # Buscar año asociado
                        year_match = re.search(r'\b(20\d{2})\b', query)
                        year = int(year_match.group(1)) if year_match else today.year
                        
                        # Primer y último día del mes
                        first_day = datetime(year, month_num, 1)
                        if month_num == 12:
                            last_day = datetime(year + 1, 1, 1) - timedelta(days=1)
                        else:
                            last_day = datetime(year, month_num + 1, 1) - timedelta(days=1)
                        
                        filters["date_from"] = first_day.strftime("%Y-%m-%d")
                        filters["date_to"] = last_day.strftime("%Y-%m-%d")
                        break
            
        return filters if filters else None
    
    @staticmethod
    def detect_country(query: str) -> Optional[str]:
        """Detecta el país mencionado en la consulta"""
        def normalize(text: str) -> str:
            import unicodedata
            text = text.lower()
            text = unicodedata.normalize('NFKD', text)
            return ''.join(c for c in text if not unicodedata.combining(c))

        query_norm = normalize(query)

        country_mapping = {
            "argentina": "Argentina",
            "méxico": "México",
            "mexico": "México",
            "colombia": "Colombia",
            "chile": "Chile",
            "perú": "Perú",
            "peru": "Perú",
            "brasil": "Brasil",
            "brazil": "Brasil",
            "uruguay": "Uruguay",
            "paraguay": "Paraguay",
            "venezuela": "Venezuela",
            "ecuador": "Ecuador",
            "bolivia": "Bolivia",
            "estados unidos": "Estados Unidos",
            "usa": "Estados Unidos",
            "eeuu": "Estados Unidos",
            "ee.uu.": "Estados Unidos",
            "ee.uu": "Estados Unidos",
            "ee uu": "Estados Unidos",
            "u.s.a.": "Estados Unidos",
            "eua": "Estados Unidos"
        }

        for key, value in country_mapping.items():
            if normalize(key) in query_norm:
                return value
        
        return None


    @staticmethod
    def extract_keywords(query: str) -> List[str]:
        """Extrae palabras clave importantes de la consulta"""
        # Lista expandida de stopwords en español
        stopwords = {"de", "la", "el", "en", "y", "a", "los", "del", "se", "las", 
                    "por", "un", "para", "con", "no", "una", "su", "al", "es", "lo",
                    "como", "más", "pero", "sus", "le", "ya", "o", "fue", "este",
                    "ha", "sí", "porque", "esta", "son", "entre", "está", "cuando",
                    "muy", "sin", "sobre", "también", "me", "hasta", "hay", "donde",
                    "han", "quien", "están", "estado", "desde", "todo", "nos", "durante",
                    "estados", "todos", "uno", "les", "ni", "contra", "otros", "fueron",
                    "ese", "eso", "había", "ante", "ellos", "e", "esto", "mí", "antes",
                    "algunos", "qué", "unos", "yo", "otro", "otras", "otra", "él", "tanto",
                    "esa", "estos", "mucho", "quienes", "nada", "muchos", "cual", "sea",
                    "poco", "ella", "estar", "haber", "estas", "estaba", "estamos", "algunas",
                    "algo", "nosotros", "mi", "mis", "tú", "te", "ti", "tu", "tus", "ellas"}
        
        # Remover fechas de las keywords
        query_clean = re.sub(r'\d{1,2}[/-]\d{1,2}[/-]\d{4}', '', query)
        query_clean = re.sub(r'\d{4}[/-]\d{1,2}[/-]\d{1,2}', '', query_clean)
        
        words = query_clean.lower().split()
        keywords = [w for w in words if w not in stopwords and len(w) > 3 and not w.isdigit()]
        return keywords

# ─── RETRIEVER PERSONALIZADO CON FILTRO DE FECHAS ───────────────────────────────
class CustomRetrieverWithDateFilter(BaseRetriever):
    """Retriever que puede filtrar por fechas además de búsqueda semántica"""
    
    vectorstore: SupabaseVectorStore
    k: int = 5
    query_processor: QueryProcessor = QueryProcessor()
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Busca documentos relevantes con posible filtro de fecha"""
        # Extraer filtros de fecha
        date_filters = self.query_processor.extract_date_filters(query)
        
        if date_filters:
            # Si hay filtros de fecha, hacer búsqueda directa en Supabase
            return self._search_by_date(query, date_filters)
        else:
            # Búsqueda semántica normal
            keywords = self.query_processor.extract_keywords(query)
            docs = self.vectorstore.similarity_search(query, k=self.k * 2)
            
            if keywords:
                docs = self._rerank_by_keywords(docs, keywords)
            
            return docs[:self.k]
    
    def _search_by_date(self, query: str, date_filters: Dict[str, Any]) -> List[Document]:
        """Busca documentos por fecha en Supabase"""
        try:
            print(f"🔍 Buscando con filtros: {date_filters}")
            
            # Construir query base
            query_builder = supabase.from_("alertas").select("*")
            
            # Aplicar filtros de fecha
            if "specific_date" in date_filters:
                # El formato en BD es YYYY-MM-DD, usar solo specific_date
                query_builder = query_builder.eq("presentation_date", date_filters["specific_date"])
                print(f"📅 Buscando fecha exacta: {date_filters['specific_date']}")
                
            elif "date_from" in date_filters and "date_to" in date_filters:
                # Búsqueda por rango
                query_builder = query_builder.gte("presentation_date", date_filters["date_from"])
                query_builder = query_builder.lte("presentation_date", date_filters["date_to"])
            
            # Aplicar filtros adicionales si hay keywords
            keywords = self.query_processor.extract_keywords(query)
            if keywords:
                # Detectar país en la consulta
                country_filter = self.query_processor.detect_country(query)

                # Aplicar filtro de país si se detectó
                if country_filter:
                    query_builder = query_builder.eq("country", country_filter)
                    print(f"🌍 Filtrando por país: {country_filter}")
            
            # Ordenar por fecha
            query_builder = query_builder.order("presentation_date", desc=True)
            
            # Ejecutar query
            print("🔄 Ejecutando consulta...")
            response = query_builder.limit(self.k * 2).execute()
            
            print(f"📊 Resultados encontrados: {len(response.data) if response.data else 0}")
            
            # Si no hay resultados con fecha exacta, buscar en un rango de ±1 día
            if not response.data and "specific_date" in date_filters:
                print("🔄 No se encontraron resultados exactos, buscando en rango de ±1 día...")
                from datetime import datetime, timedelta
                
                date_obj = datetime.strptime(date_filters["specific_date"], "%Y-%m-%d")
                date_before = (date_obj - timedelta(days=1)).strftime("%Y-%m-%d")
                date_after = (date_obj + timedelta(days=1)).strftime("%Y-%m-%d")
                
                query_builder = supabase.from_("alertas").select("*")
                query_builder = query_builder.gte("presentation_date", date_before)
                query_builder = query_builder.lte("presentation_date", date_after)
                
                if "argentina" in [k.lower() for k in keywords]:
                    query_builder = query_builder.eq("country", "Argentina")
                
                response = query_builder.order("presentation_date", desc=True).execute()
                print(f"📊 Resultados en rango ampliado: {len(response.data) if response.data else 0}")
            
            # Convertir a Documents
            documents = []
            for item in (response.data or []):
                content = f"{item.get('title', '')}\n\n{item.get('description', '')}"
                
                metadata = {
                    "id": item.get("id"),
                    "title": item.get("title", "Sin título"),
                    "created_at": item.get("created_at"),
                    "presentation_date": item.get("presentation_date"),
                    "category": item.get("category", ""),
                    "country": item.get("country", ""),
                    "institution": item.get("institution", ""),
                    "source_url": item.get("source_url", "")
                }
                
                documents.append(Document(page_content=content, metadata=metadata))
            
            return documents[:self.k]
            
        except Exception as e:
            print(f"❌ Error en búsqueda por fecha: {str(e)}")
            import traceback
            traceback.print_exc()
            # Fallback a búsqueda semántica
            return self.vectorstore.similarity_search(query, k=self.k)
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Versión asíncrona"""
        return self._get_relevant_documents(query)
    
    def _rerank_by_keywords(self, docs: List[Document], keywords: List[str]) -> List[Document]:
        """Reordena documentos basándose en presencia de keywords"""
        scored_docs = []
        
        for doc in docs:
            keyword_score = 0
            content_lower = doc.page_content.lower()
            
            for keyword in keywords:
                if keyword in content_lower:
                    keyword_score += content_lower.count(keyword)
            
            scored_docs.append((doc, keyword_score))
        
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs]

# ─── CADENA CONVERSACIONAL MEJORADA ─────────────────────────────────────────────
class ImprovedChatbot:
    def __init__(self):
        # Configurar LLM
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.1,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
        # Configurar vectorstore
        self.vectorstore = self._get_enhanced_vectorstore()
        
        # Configurar retriever personalizado con filtro de fechas
        self.retriever = CustomRetrieverWithDateFilter(
            vectorstore=self.vectorstore,
            k=5
        )
        
        # Configurar memoria
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Configurar prompt
        self.prompt = PromptTemplate(
            template=SYSTEM_PROMPT,
            input_variables=["context", "chat_history", "question"]
        )
        
        # Crear cadena conversacional
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.prompt},
            return_source_documents=True,
            verbose=True
        )
    
    def _get_enhanced_vectorstore(self):
        """Crea un vectorstore con capacidades mejoradas"""
        return SupabaseVectorStore(
            client=supabase,
            embedding=embeddings,
            table_name="alertas",
            query_name="match_alertas_avanzado"  # Usar función básica si la avanzada no existe
        )
    
    def chat(self, question: str) -> Dict[str, Any]:
        """Procesa una pregunta y retorna respuesta con fuentes"""
        print(f"\n🤔 Pregunta: {question}")
        print("\n💭 Procesando...")
        
        try:
            # Detectar si es una consulta por fecha
            date_filters = QueryProcessor.extract_date_filters(question)
            if date_filters:
                print(f"📅 Filtros de fecha detectados: {date_filters}")
            
            # Obtener respuesta
            result = self.chain.invoke({"question": question})
            
            # Formatear respuesta
            response = {
                "answer": result.get("answer", "No se pudo generar una respuesta"),
                "sources": []
            }
            
            # Procesar documentos fuente
            if "source_documents" in result:
                print("\n📚 Fuentes consultadas:")
                for i, doc in enumerate(result["source_documents"]):
                    # Usar presentation_date como fecha principal
                    date_value = (doc.metadata.get("presentation_date") or 
                                doc.metadata.get("created_at") or 
                                "Fecha no disponible")
                    
                    title = doc.metadata.get("title", "")
                    if not title and doc.page_content:
                        title = doc.page_content.split('\n')[0][:100]
                    
                    source_info = {
                        "index": i + 1,
                        "title": title or "Sin título",
                        "date": date_value,
                        "category": doc.metadata.get("category", ""),
                        "country": doc.metadata.get("country", ""),
                        "institution": doc.metadata.get("institution", ""),
                        "id": doc.metadata.get("id", ""),
                        "source_url": doc.metadata.get("source_url", ""),
                        "excerpt": doc.page_content[:200] + "..."
                    }
                    response["sources"].append(source_info)
                    
                    print(f"\n[{i+1}] {source_info['title']}")
                    print(f"    📅 {source_info['date']}")
                    if source_info['source_url']:
                        print(f"    🔗 {source_info['source_url']}")
                    print(f"    🌍 {source_info['country']}")
                    if source_info['category']:
                        print(f"    📁 {source_info['category']}")
            
            return response
            
        except Exception as e:
            print(f"\n❌ Error procesando la pregunta: {str(e)}")
            return {
                "answer": "Lo siento, ocurrió un error al procesar tu pregunta. Por favor, intenta reformularla.",
                "sources": []
            }
    
    def search_by_date_range(self, start_date: str, end_date: str, country: Optional[str] = None) -> List[Dict[str, Any]]:
        """Busca alertas en un rango de fechas específico"""
        try:
            print(f"🔍 Buscando alertas entre {start_date} y {end_date}")
            
            # Construir query (para campos de texto)
            query = supabase.from_("alertas") \
                .select("id, title, description, presentation_date, created_at, category, country, institution") \
                .filter("presentation_date", "gte", start_date) \
                .filter("presentation_date", "lte", end_date)
            
            # Filtrar por país si se especifica
            if country:
                query = query.eq("country", country)
            
            # Ordenar por fecha de presentación
            query = query.order("presentation_date", desc=True)
            
            # Ejecutar query
            response = query.execute()
            
            # Procesar resultados
            results = []
            for item in (response.data or []):
                normalized = {
                    'id': item.get('id'),
                    'title': item.get('title', 'Sin título'),
                    'description': item.get('description', '')[:200] + "...",
                    'presentation_date': item.get('presentation_date', 'Fecha no disponible'),
                    'created_at': item.get('created_at', ''),
                    'category': item.get('category', 'Sin categoría'),
                    'country': item.get('country', 'Sin país'),
                    'institution': item.get('institution', 'Sin institución')
                }
                results.append(normalized)
            
            return results
                
        except Exception as e:
            print(f"❌ Error en búsqueda por fecha: {str(e)}")
            return []
    
    def check_date_format(self):
        """Verifica el formato de fechas en la base de datos"""
        try:
            # Obtener algunas muestras de presentation_date
            response = supabase.from_("alertas") \
                .select("presentation_date") \
                .not_.is_("presentation_date", "null") \
                .limit(5) \
                .execute()
            
            if response.data:
                print("\n📅 Formato de fechas en la base de datos:")
                for record in response.data:
                    print(f"  - {record['presentation_date']}")
                return response.data[0]['presentation_date'] if response.data else None
        except Exception as e:
            print(f"❌ Error verificando formato de fechas: {str(e)}")
            return None

    def generate_summary(self, topic: str, date_from: Optional[str] = None, date_to: Optional[str] = None) -> str:
        """Genera un resumen ejecutivo sobre un tema específico"""
        try:
            # Si hay fechas, buscar por rango
            if date_from and date_to:
                alerts = self.search_by_date_range(date_from, date_to)
                
                if not alerts:
                    return f"No se encontraron alertas sobre '{topic}' entre {date_from} y {date_to}."
                
                # Crear contexto con las alertas encontradas
                context = "\n\n".join([
                    f"Alerta: {alert['title']}\n"
                    f"Fecha: {alert['presentation_date']}\n"
                    f"País: {alert['country']}\n"
                    f"Institución: {alert['institution']}\n"
                    f"Contenido: {alert['description']}"
                    for alert in alerts[:20]  # Limitar a 20 alertas
                ])
            else:
                # Búsqueda semántica normal
                docs = self.vectorstore.similarity_search(topic, k=10)
                
                if not docs:
                    return "No se encontraron alertas relacionadas con este tema."
                
                context = "\n\n".join([
                    f"Alerta: {doc.metadata.get('title', 'Sin título')}\n"
                    f"Fecha: {doc.metadata.get('presentation_date', 'N/A')}\n"
                    f"Contenido: {doc.page_content[:500]}..."
                    for doc in docs
                ])
            
            # Crear prompt específico para resumen
            summary_prompt = f"""
            Basándote en las siguientes alertas regulatorias sobre "{topic}", genera un resumen ejecutivo estructurado:
            
            {context}
            
            El resumen debe incluir:
            1. **Principales regulaciones identificadas**: Lista las regulaciones más importantes
            2. **Tendencias observadas**: Patrones o tendencias en las regulaciones
            3. **Países/Instituciones involucradas**: Menciona qué países e instituciones están emitiendo estas alertas
            4. **Impacto potencial**: Cómo estas regulaciones podrían afectar al sector
            5. **Recomendaciones clave**: Acciones sugeridas basadas en las alertas
            
            Formato el resumen de manera clara y profesional.
            """
            
            response = self.llm.predict(summary_prompt)
            return response
            
        except Exception as e:
            return f"Error al generar el resumen: {str(e)}"

# ─── FUNCIONES AUXILIARES ────────────────────────────────────────────────────────
def normalize_date(date_str: str) -> str:
    """Convierte una fecha en distintos formatos a YYYY-MM-DD"""
    parts = re.split(r'[\/\.-]', date_str)
    if len(parts[0]) == 4:
        year, month, day = parts
    else:
        day, month, year = parts
    return f"{year}-{month.zfill(2)}-{day.zfill(2)}"


def parse_date_command(command: str) -> Optional[tuple]:
    """Parsea comandos de búsqueda por fecha"""
    # Buscar patrones de fecha, aceptando distintos separadores
    patterns = [
        r'desde\s+(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{4})\s+hasta\s+(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{4})',
        r'entre\s+(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{4})\s+y\s+(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{4})',
        r'del\s+(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{4})\s+al\s+(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{4})',
        r'(\d{4}[\/\.\-]\d{1,2}[\/\.\-]\d{1,2})\s+a[l]?\s+(\d{4}[\/\.\-]\d{1,2}[\/\.\-]\d{1,2})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, command, re.IGNORECASE)
        if match:
            date1, date2 = match.groups()
            date1 = normalize_date(date1)
            date2 = normalize_date(date2)
            return (date1, date2)
    
    return None

def test_connection():
    """Prueba la conexión con Supabase"""
    try:
        response = supabase.from_("alertas").select("count", count="exact").execute()
        print(f"✅ Conexión exitosa. Total de alertas: {response.count}")
        return True
    except Exception as e:
        print(f"❌ Error de conexión: {str(e)}")
        return False

def diagnose_database_structure():
    """Diagnostica la estructura de la base de datos"""
    print("\n🔍 Diagnóstico de la base de datos:")
    print("=" * 50)
    
    try:
        # 1. Verificar estructura de la tabla
        response = supabase.from_("alertas").select("*").limit(1).execute()
        
        if response.data and len(response.data) > 0:
            sample = response.data[0]
            print("\n📋 Campos disponibles en la tabla 'alertas':")
            for field, value in sample.items():
                field_type = type(value).__name__
                if field == 'embedding':
                    print(f"  - {field}: vector ({len(value) if value else 0} dimensiones)")
                else:
                    print(f"  - {field}: {field_type} = {str(value)[:50]}...")
        
        # 2. Contar registros totales
        count_response = supabase.from_("alertas").select("count", count="exact").execute()
        print(f"\n📊 Total de registros: {count_response.count}")
        
        # 3. Verificar registros con embeddings
        embedding_count = supabase.from_("alertas").select("count", count="exact").not_.is_("embedding", "null").execute()
        print(f"📊 Registros con embeddings: {embedding_count.count}")
        
        # 4. Verificar distribución por fechas
        print("\n📅 Distribución por presentation_date:")
        date_sample = supabase.from_("alertas") \
            .select("presentation_date") \
            .not_.is_("presentation_date", "null") \
            .order("presentation_date", desc=True) \
            .limit(10) \
            .execute()
        
        if date_sample.data:
            for record in date_sample.data:
                print(f"  - {record['presentation_date']}")
        else:
            print("  ❌ No hay datos de presentation_date")
            
    except Exception as e:
        print(f"❌ Error en diagnóstico: {str(e)}")

# ─── INTERFAZ DE USUARIO ─────────────────────────────────────────────────────────
def main():
    print("🤖 Chatbot de Alertas Regulatorias - Versión Mejorada con Búsqueda por Fecha")
    print("=" * 70)
    
    # Probar conexión
    if not test_connection():
        print("\n⚠️  No se pudo conectar a la base de datos.")
        return
    
    # Ejecutar diagnóstico
    print("\n¿Deseas ejecutar un diagnóstico de la base de datos? (s/n): ", end="")
    if input().lower() in ['s', 'si', 'yes', 'y']:
        diagnose_database_structure()
    
    # Inicializar chatbot
    try:
        chatbot = ImprovedChatbot()
        print("\n✅ Chatbot inicializado correctamente")
    except Exception as e:
        print(f"\n❌ Error inicializando chatbot: {str(e)}")
        return
    
    print("\nComandos disponibles:")
    print("- 'resumen [tema]': Genera un resumen ejecutivo")
    print("- 'resumen [tema] desde DD/MM/YYYY hasta DD/MM/YYYY': Resumen con rango de fechas")
    print("- 'buscar desde DD/MM/YYYY hasta DD/MM/YYYY': Busca por rango de fechas")
    print("- 'diagnostico': Ejecuta diagnóstico de la base de datos")
    print("- 'exit': Salir")
    print("\nEjemplos de búsqueda:")
    print("- 'alertas de argentina del 14/02/2025'")
    print("- 'regulaciones entre 01/01/2025 y 31/01/2025'")
    print("- 'alertas de febrero 2025'")
    print("- 'últimas alertas de la semana pasada'")
    print("\n")
    
    while True:
        user_input = input("\n💬 Tu pregunta: ").strip()
        
        if user_input.lower() in ["exit", "salir", "quit"]:
            print("👋 ¡Hasta luego!")
            break
        
        elif user_input.lower() == "diagnostico":
            diagnose_database_structure()
        
        elif user_input.lower().startswith("resumen"):
            # Extraer tema y posibles fechas
            command_parts = user_input[7:].strip()
            
            # Buscar si hay fechas en el comando
            date_range = parse_date_command(command_parts)
            
            if date_range:
                # Extraer el tema (lo que está antes de las fechas)
                topic = re.split(r'desde|hasta|entre', command_parts)[0].strip()
                if not topic:
                    topic = "alertas regulatorias"
                
                print(f"\n📊 Generando resumen de '{topic}' entre {date_range[0]} y {date_range[1]}...")
                summary = chatbot.generate_summary(topic, date_range[0], date_range[1])
                print(f"\n{summary}")
            else:
                # Resumen sin fechas
                topic = command_parts
                if topic:
                    print(f"\n📊 Generando resumen ejecutivo sobre '{topic}'...")
                    summary = chatbot.generate_summary(topic)
                    print(f"\n{summary}")
                else:
                    print("❌ Por favor especifica un tema para el resumen")
        
        # Verificar que hay datos para la fecha buscada
        elif user_input.lower().startswith("verificar fecha"):
            fecha = input("Ingresa la fecha a verificar (YYYY-MM-DD): ")
            response = supabase.from_("alertas") \
                .select("id, title, presentation_date, country") \
                .eq("presentation_date", fecha) \
                .execute()
            
            if response.data:
                print(f"\n✅ Se encontraron {len(response.data)} alertas para {fecha}:")
                for r in response.data[:5]:
                    print(f"  - {r['title'][:50]}... ({r['country']})")
            else:
                print(f"❌ No hay alertas para la fecha {fecha}")

        elif user_input.lower().startswith("buscar"):
            # Buscar fechas en el comando
            date_range = parse_date_command(user_input)
            
            if date_range:
                start_date, end_date = date_range
                
                # Buscar país si se menciona
                country = None
                if "argentina" in user_input.lower():
                    country = "Argentina"
                elif "mexico" in user_input.lower() or "méxico" in user_input.lower():
                    country = "México"
                elif "colombia" in user_input.lower():
                    country = "Colombia"
                
                print(f"\n🔍 Buscando alertas entre {start_date} y {end_date}...")
                if country:
                    print(f"    🌍 País: {country}")
                
                results = chatbot.search_by_date_range(start_date, end_date, country)
                
                if results:
                    print(f"\n📋 Se encontraron {len(results)} alertas:")
                    for i, alert in enumerate(results[:15]):  # Mostrar máximo 15
                        print(f"\n[{i+1}] {alert['title']}")
                        print(f"    📅 {alert['presentation_date']}")
                        print(f"    🌍 {alert['country']} - {alert['institution']}")
                        print(f"    📁 {alert['category']}")
                        if alert['description']:
                            print(f"    📝 {alert['description'][:100]}...")
                    
                    if len(results) > 15:
                        print(f"\n... y {len(results) - 15} alertas más")
                else:
                    print("❌ No se encontraron alertas en ese rango de fechas")
            else:
                print("❌ No se detectó un rango de fechas válido. Usa el formato: 'buscar desde DD/MM/YYYY hasta DD/MM/YYYY'")
        
        else:
            # Pregunta normal al chatbot
            response = chatbot.chat(user_input)
            print(f"\n🤖 Respuesta:\n{response['answer']}")

if __name__ == "__main__":
    main()
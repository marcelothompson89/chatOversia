import os
from supabase import create_client
from openai import OpenAI
from dotenv import load_dotenv

# ─── CONFIGURACIÓN ───────────────────────────────────────────────────────────────
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

# ─── CLIENTES ──────────────────────────────────────────────────────────────────
openai = OpenAI()
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def ingest_embeddings():
    print("🚀 Iniciando carga de embeddings para todos los registros...\n")
    # 1) Leer todos los registros (que no tengan ya un embedding)
    resp = supabase.from_("alertas") \
                   .select("id, title, description") \
                   .filter("embedding", "is", "null") \
                   .execute()
    alertas = resp.data or []

    if not alertas:
        print("⚠️  No hay registros sin embedding.")
        return

    print(f"🔎 Total de registros a procesar: {len(alertas)}")

    # 2) Iterar sobre cada alerta
    for alerta in alertas:
        texto = f"{alerta['title']}\n\n{alerta['description']}"
        try:
            emb_resp = openai.embeddings.create(
                model="text-embedding-3-small",
                input=texto
            )
            vector = emb_resp.data[0].embedding
        except Exception as e:
            print(f"❌ Error generando embedding para id={alerta['id']}: {e}")
            continue

        # 3) Actualizar en Supabase
        try:
            upd = supabase.from_("alertas") \
                          .update({"embedding": vector}) \
                          .eq("id", alerta["id"]) \
                          .execute()

            status = getattr(upd, "status_code", None)
            if status == 200:
                print(f"✅ Embedding guardado para id={alerta['id']}")
            else:
                print(f"❌ Falló update para id={alerta['id']} - status {status}")
        except Exception as e:
            print(f"❌ Error actualizando id={alerta['id']}: {e}")
            continue

    print("\n🎉 Proceso completado.")

if __name__ == "__main__":
    ingest_embeddings()
import os
from dotenv import load_dotenv, find_dotenv

print("🧭 Current working directory:", os.getcwd())

found_path = find_dotenv()
print("📁 python-dotenv found this .env:", found_path)

forced_path = os.path.join(os.path.dirname(__file__), ".env")
print("⚙️ Forcing dotenv path:", forced_path)

load_dotenv(dotenv_path=forced_path, override=True)
print("🔑 Loaded key:", os.getenv("OPENAI_API_KEY"))

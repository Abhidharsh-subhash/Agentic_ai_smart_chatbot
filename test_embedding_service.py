import sys

sys.path.insert(0, ".")

from app.services.embeddings import embedding_service

print("Testing embedding service...")
print(f"Index file exists: {embedding_service.index_file.exists()}")
print(f"Index dir: {embedding_service.index_dir}")

try:
    result = embedding_service.get_available_documents()
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()

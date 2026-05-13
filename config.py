from pathlib import Path

# Resolving the project root relative to this file
PROJECT_ROOT = Path(__file__).parent

# Defining data cache paths
DATA_CACHE_DIR = PROJECT_ROOT / "data" / "cache"
RAW_COCKTAILS_PATH = DATA_CACHE_DIR / "raw_cocktails.json"

# Defining embedding index paths
EMBEDDINGS_INDEX_DIR = PROJECT_ROOT / "embeddings" / "index"
EMBEDDINGS_PATH = EMBEDDINGS_INDEX_DIR / "embeddings.npy"
METADATA_PATH = EMBEDDINGS_INDEX_DIR / "metadata.pkl"

# Specifying the sentence-transformers model to use for embedding
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Setting the TheCocktailDB API base URL
API_BASE_URL = "https://www.thecocktaildb.com/api/json/v1/1"

# Defining the default number of results returned per query
DEFAULT_TOP_K = 5

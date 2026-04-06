# ---
# deploy: false
# ---

# # Vector similarity search with sentence-transformers and pgvector on Neon
#
# This example shows how to build a **semantic search** pipeline on Modal:
# embed a corpus of sentences with a GPU-accelerated sentence-transformer model,
# store the resulting vectors in a [pgvector](https://github.com/pgvector/pgvector)
# table on [Neon](https://neon.tech), and run cosine-similarity queries against
# them — all from a single Python file.
#
# Along the way you'll see how Modal makes it easy to:
#
# * install Python packages into a custom container image,
# * run a function on a GPU (even a small T4 is plenty for embedding),
# * inject database credentials via a [Modal Secret](https://modal.com/docs/guide/secrets),
# * and wire several remote functions together from a `local_entrypoint`.
#
# ## Prerequisites
#
# Before running this example you need:
#
# 1. A [Neon](https://neon.tech) Postgres database with the `pgvector` extension
#    enabled.  You can enable it with `CREATE EXTENSION IF NOT EXISTS vector;`
#    in the Neon SQL editor.
# 2. A Modal Secret named `neonpgvector` that contains a single key
#    `DATABASE_URL` whose value is your Neon connection string, e.g.
#    `postgresql://alice:secret@ep-xyz.us-east-2.aws.neon.tech/neondb?sslmode=require`.
#    Create it at https://modal.com/secrets.

import os

import modal

# ## Defining the container image
#
# Modal functions run inside containers built from an `Image`.  We start from a
# slim Debian base and install two packages:
#
# * `sentence-transformers` — provides the `all-MiniLM-L6-v2` model and wraps
#   the heavy `torch` / `transformers` stack.
# * `psycopg2-binary` — a self-contained Postgres client that needs no system
#   libraries at runtime (unlike the source-compiled `psycopg2`).

image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "sentence-transformers==3.4.1",
    "psycopg2-binary==2.9.10",
)

app = modal.App("example-vector-similarity-search", image=image)

# ## The embed function
#
# `embed` is the only function that needs a GPU.  It loads `all-MiniLM-L6-v2`
# once per container (the model is small — about 90 MB — but loading it still
# takes a few seconds) and then encodes a list of strings into 384-dimensional
# float32 vectors.
#
# We request a T4: it's the most affordable GPU on Modal and more than fast
# enough for this model.  The return type is a plain Python list so the result
# can be serialised and sent back to the caller without any extra dependencies.


@app.function(gpu="T4")
def embed(texts: list[str]) -> list[list[float]]:
    from sentence_transformers import SentenceTransformer

    # Load the model onto the GPU.  Modal caches the container filesystem
    # between invocations, so on warm starts the model is already present.
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

    # `encode` returns a numpy array; `.tolist()` converts it to a plain
    # Python list of lists which is JSON-serialisable and easy to work with
    # downstream.
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()


# ## The index function
#
# `index` owns the "write" side of the pipeline.  It:
#
# 1. Embeds a small hardcoded corpus with `embed`.
# 2. Opens a connection to the Neon database using the `DATABASE_URL`
#    environment variable injected by our Modal Secret.
# 3. Creates the `pgvector` extension and a `documents` table if they don't
#    already exist.
# 4. Truncates the table so re-running `index` always gives a clean slate,
#    then bulk-inserts all (text, embedding) pairs.
#
# The corpus is intentionally varied — space, machine learning, cooking, and
# climate — so that the demo queries later can show meaningful nearest-neighbour
# behaviour.

CORPUS = [
    # Space exploration
    "The James Webb Space Telescope captures infrared light from the earliest galaxies.",
    "Astronauts on the International Space Station conduct microgravity experiments daily.",
    "SpaceX Starship is designed to carry humans to the Moon and eventually to Mars.",
    "Black holes warp spacetime so severely that not even light can escape their gravity.",
    # Machine learning
    "Transformer models use self-attention to capture long-range dependencies in text.",
    "Gradient descent iteratively adjusts model weights to minimise a loss function.",
    "Transfer learning lets you fine-tune a pretrained model on a small domain dataset.",
    "Vector embeddings map words and sentences to points in a high-dimensional space.",
    # Climate and environment
    "Rising ocean temperatures are causing widespread coral bleaching across reef systems.",
    "Renewable energy sources like solar and wind now undercut fossil fuels on cost.",
    "Reforestation projects sequester carbon and restore biodiversity in degraded land.",
    # Cooking and food
    "Maillard browning gives seared steak its complex savoury crust and rich aroma.",
    "Sourdough fermentation relies on wild yeast and lactic-acid bacteria for leavening.",
    "Emulsification lets olive oil and lemon juice combine into a stable vinaigrette.",
]


@app.function(secrets=[modal.Secret.from_name("neonpgvector")])
def index() -> None:
    import psycopg2

    print(f"Embedding {len(CORPUS)} documents...")
    # Call embed remotely — it runs on a GPU container while index runs on CPU.
    vectors = embed.remote(CORPUS)

    database_url = os.environ["DATABASE_URL"]
    conn = psycopg2.connect(database_url)
    conn.autocommit = True
    cur = conn.cursor()

    # Enable pgvector.  The IF NOT EXISTS guard makes this idempotent.
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # all-MiniLM-L6-v2 produces 384-dimensional vectors.
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id   SERIAL PRIMARY KEY,
            text TEXT NOT NULL,
            embedding vector(384) NOT NULL
        );
        """
    )

    # Wipe any previous run so the table is always in a known state.
    cur.execute("TRUNCATE TABLE documents;")

    # Bulk-insert all rows.  We use `%s` placeholders and pass the embedding
    # as a string in pgvector's `[f1, f2, ...]` literal format.
    for text, vector in zip(CORPUS, vectors):
        vector_literal = "[" + ",".join(str(v) for v in vector) + "]"
        cur.execute(
            "INSERT INTO documents (text, embedding) VALUES (%s, %s);",
            (text, vector_literal),
        )

    cur.close()
    conn.close()
    print("Indexed", len(CORPUS), "documents.")


# ## The search function
#
# `search` is the "read" side.  Given a query string it:
#
# 1. Embeds the query with `embed` (same model, same vector space as the corpus).
# 2. Runs a pgvector cosine-distance query that sorts all rows by distance and
#    returns the five closest.
#
# Cosine distance (`<=>`) measures the angle between two vectors, which makes
# it robust to differences in text length.  A distance of 0 means identical
# direction; 2 means maximally opposite.


@app.function(secrets=[modal.Secret.from_name("neonpgvector")])
def search(query: str, top_k: int = 5) -> list[tuple[str, float]]:
    import psycopg2

    # Embed the query on the GPU using the same model as the corpus.
    (query_vector,) = embed.remote([query])
    query_literal = "[" + ",".join(str(v) for v in query_vector) + "]"

    database_url = os.environ["DATABASE_URL"]
    conn = psycopg2.connect(database_url)
    cur = conn.cursor()

    # `<=>` is pgvector's cosine-distance operator.  We return both the text
    # and the distance so the caller can inspect how similar each result is.
    cur.execute(
        """
        SELECT text, (embedding <=> %s::vector) AS distance
        FROM   documents
        ORDER  BY distance
        LIMIT  %s;
        """,
        (query_literal, top_k),
    )

    results = [(row[0], float(row[1])) for row in cur.fetchall()]
    cur.close()
    conn.close()
    return results


# ## Putting it all together
#
# The `local_entrypoint` is the CLI entry point when you run
# `modal run 06_gpu_and_ml/vector_similarity_search.py`.
# It first indexes the corpus, then fires off a handful of queries that span
# different topics so you can see how well semantic search retrieves thematically
# related sentences even when none of the exact words appear in the query.


@app.local_entrypoint()
def main() -> None:
    print("=== Indexing corpus ===")
    index.remote()

    demo_queries = [
        "exploring outer space and distant planets",
        "deep learning and neural network training",
        "global warming and carbon emissions",
        "baking bread at home",
    ]

    print("\n=== Running demo queries ===")
    for query in demo_queries:
        print(f'\nQuery: "{query}"')
        results = search.remote(query)
        for rank, (text, distance) in enumerate(results, start=1):
            print(f"  {rank}. [dist={distance:.4f}] {text}")

"""
This project uses PostgreSQL to store fact-data as-well-as
vector embeddings. This file manages the database, from
query execution to creating the database.

This file `data_engine.py` is responsible for the following:
* Database creation (from csv)
* Embedding generation using an embedding model
* Database querying API, both embeddings and facts
"""

import os
import re
import psycopg
import warnings
import pandas as pd
import logging

from dotenv import load_dotenv
from psycopg.abc import QueryNoTemplate
from tqdm.auto import tqdm

from models import embedding_model

# Pre-retrieval query refinement prompts
QUERY_REFINE_SYSTEM_PROMPT = (
    "You are a search query optimizer for a movie database. "
    "Rewrite the user's conversational message into a concise, keyword-rich "
    "search string optimized for semantic similarity search against movie descriptions. "
    "Include inferred genres, themes, mood, and relevant keywords. "
    "Output ONLY the refined query string, nothing else. No explanations."
)

# Contextual enrichment prompt template for DB creation
ENRICHMENT_PROMPT_TEMPLATE = (
    "Given the following movie information, write 1-2 sentences describing the "
    "movie's themes, tone, emotional appeal, and what kind of viewer would enjoy it. "
    "Be concise and descriptive. Output ONLY the sentences, nothing else.\n\n"
    "Title: {title}\n"
    "Tagline: {tagline}\n"
    "Overview: {overview}\n"
    "Genres: {genres}\n"
    "Keywords: {keywords}\n"
    "Director: {director}\n"
    "Cast: {cast}\n"
)


# connect to db
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s::%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

# db stuff
DB_URL = os.getenv("DB_URL")
assert isinstance(DB_URL, str), "Invalid database URL"


# db utils
def _establish_connection(db_url: str):
    """
    Connect to database at URL.
    Returns the connection (conn) and cursor (cur) once connectionis
    established.

    Args:
        db_url (str): database URL
    Returns:
        tuple: conn, cur
    """
    conn = psycopg.connect(db_url)
    cur = conn.cursor()

    logger.info(f"Connected to db: {conn}")

    return conn, cur


def _close_conn(conn):
    """Close database connection (conn)."""
    conn.close()
    logger.info("Connection closed.")


# data stuff
data_path = "data/clean_data.csv"
BATCH_SIZE = 64


def _load_enrichment_model():
    """
    Hot-load a small LLM for generating contextual enrichment text.
    Returns (tokenizer, model, device) tuple. Caller is responsible
    for deleting these when done.

    Model used: Qwen2.5-1.5B-Instruct
    """
    from transformers import pipeline
    from utils import auto_device_map

    device = auto_device_map()
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"

    logger.info(f"Loading enrichment model: {model_id}")
    # tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id,
    #     trust_remote_code=True,
    #     device_map="auto",
    #     torch_dtype=torch.bfloat16,
    # )
    # model.eval()
    pipe = pipeline("text-generation", model_id, device_map="auto")
    logger.info("Enrichment model loaded.")
    return pipe


def _generate_enrichment_batch(pipe, batch_df) -> list[str]:
    """
    Generate 1-2 contextual enrichment sentences for a batch of movie rows
    using a Hugging Face pipeline. Prompts are batched for efficiency.

    Args:
        pipe: Hugging Face text-generation or text2text-generation pipeline
        batch_df: DataFrame slice of movie rows

    Returns:
        list[str]: enrichment sentences, one per row, in input order
    """

    prompts = []
    for _, row in batch_df.iterrows():
        prompt = ENRICHMENT_PROMPT_TEMPLATE.format(
            title=row["title"],
            tagline=row["tagline"],
            overview=row["overview"],
            genres=row["genres"].replace("|", ", "),
            keywords=row["keywords"].replace("|", ", "),
            director=row["director"],
            cast=row["top_cast"].replace("|", ", "),
        )

        # If using chat-style models, flatten into a single prompt string
        full_prompt = (
            "You are a concise movie analyst.\n\n"
            f"{prompt}\n\n"
            "Write 1-2 concise enrichment sentences:"
        )
        prompts.append(full_prompt)

    # Run batched inference
    outputs = pipe(
        prompts,
        max_new_tokens=100,
        batch_size=len(prompts),  # allows batching
        truncation=True,
    )

    enrichments = []

    for output in outputs:
        # pipeline returns different shapes depending on task
        if isinstance(output, list):
            output = output[0]

        # text-generation pipeline
        if "generated_text" in output:
            text = output["generated_text"]
        # text2text-generation pipeline
        elif "generated_text" in output:
            text = output["generated_text"]
        elif "summary_text" in output:
            text = output["summary_text"]
        else:
            text = str(output)

        # Remove prompt if model echoes it
        enrichment = text.strip()
        enrichments.append(enrichment)

    return enrichments


def _create_db(enrich_batch_size: int = 8):
    """
    Create/initialize the database. Creates a table `movies`
    and fills it with all data form `data/data_clean.csv`.
    Table primary key: `tmdb_id` (numeric/INT). If some row
    already exists with tmdb_id, its overwritten using fact
    data from csv.

    Uses contextual retrieval: a small LLM generates 1-2
    enrichment sentences per movie before embedding, producing
    richer vector representations.

    Args:
        enrich_batch_size (int): number of movies to enrich per
            LLM forward pass. Higher values are faster but use
            more memory. Default: 8.
    """
    import torch
    import gc

    conn, cur = _establish_connection(DB_URL)  # pyright: ignore

    # create schema if not alr
    query = """
    CREATE TABLE IF NOT EXISTS movies (
        tmdb_id INT PRIMARY KEY,
        title TEXT,
        tagline TEXT,
        overview TEXT,
        release_date DATE,
        runtime_mins INT,
        certification TEXT,
        genres TEXT[],
        keywords TEXT[],
        spoken_languages TEXT[],
        origin_country TEXT,
        collection TEXT,
        director TEXT,
        top_cast TEXT[],
        production_companies TEXT[],
        budget BIGINT,
        revenue BIGINT,
        vote_average FLOAT,
        vote_count INT,
        popularity FLOAT,
        embedding vector(768)
    );
    """
    logger.info("Creating table `movies`.")
    cur.execute(query)

    # load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    df.fillna("", inplace=True)

    # generate contextual enrichment in batches
    logger.info(
        f"Phase 1: Generating contextual enrichment with LLM "
        f"(batch_size={enrich_batch_size})..."
    )
    pipe = _load_enrichment_model()

    enrichments = []
    for i in tqdm(
        range(0, len(df), enrich_batch_size),
        desc="Enriching",
        unit="batch",
    ):
        batch_df = df.iloc[i : i + enrich_batch_size]
        batch_enrichments = _generate_enrichment_batch(
            pipe, batch_df
        )
        enrichments.extend(batch_enrichments)

    # free enrichment model memory
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    logger.info("Enrichment model unloaded. Memory freed.")

    # embed enriched texts and write to DB
    logger.info("Phase 2: Embed enriched texts and write to DB.")
    for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Embedding", unit="batch"):
        batch_df = df.iloc[i : i + BATCH_SIZE]
        batch_enrichments = enrichments[i : i + BATCH_SIZE]

        texts = [
            f"{enrichment} "
            f"Title: {row['title']} | "
            f"Genres: {row['genres'].replace('|', ', ')} | "
            f"Keywords: {row['keywords'].replace('|', ', ')} | "
            f"Director: {row['director']} | "
            f"Cast: {row['top_cast'].replace('|', ', ')} | "
            f"Plot: {row['overview']} {row['tagline']}"
            for (_, row), enrichment in zip(batch_df.iterrows(), batch_enrichments)
        ]
        embeddings = embedding_model.embed(texts).cpu().tolist()

        for (_, row), embedding in zip(batch_df.iterrows(), embeddings):
            cur.execute(
                """
                INSERT INTO movies (
                    tmdb_id,
                    title,
                    tagline,
                    overview,
                    release_date,
                    runtime_mins,
                    certification,
                    genres,
                    keywords,
                    spoken_languages,
                    origin_country,
                    collection,
                    director,
                    top_cast,
                    production_companies,
                    budget,
                    revenue,
                    vote_average,
                    vote_count,
                    popularity,
                    embedding
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (tmdb_id)
                DO UPDATE SET
                    title = EXCLUDED.title,
                    tagline = EXCLUDED.tagline,
                    overview = EXCLUDED.overview,
                    release_date = EXCLUDED.release_date,
                    runtime_mins = EXCLUDED.runtime_mins,
                    certification = EXCLUDED.certification,
                    genres = EXCLUDED.genres,
                    keywords = EXCLUDED.keywords,
                    spoken_languages = EXCLUDED.spoken_languages,
                    origin_country = EXCLUDED.origin_country,
                    collection = EXCLUDED.collection,
                    director = EXCLUDED.director,
                    top_cast = EXCLUDED.top_cast,
                    production_companies = EXCLUDED.production_companies,
                    budget = EXCLUDED.budget,
                    revenue = EXCLUDED.revenue,
                    vote_average = EXCLUDED.vote_average,
                    vote_count = EXCLUDED.vote_count,
                    popularity = EXCLUDED.popularity,
                    embedding = EXCLUDED.embedding
                ;
                """,
                (
                    row["tmdb_id"],
                    row["title"],
                    row["tagline"],
                    row["overview"],
                    row["release_date"],
                    row["runtime_mins"],
                    row["certification"],
                    row["genres"].split("|"),  # pyright: ignore
                    row["keywords"].split("|"),  # pyright: ignore
                    row["spoken_languages"].split("|"),  # pyright: ignore
                    row["origin_country"],
                    row["collection"],
                    row["director"],
                    row["top_cast"].split("|"),  # pyright: ignore
                    row["production_companies"].split("|"),  # pyright: ignore
                    row["budget"],
                    row["revenue"],
                    row["vote_average"],
                    row["vote_count"],
                    row["popularity"],
                    embedding,
                ),
            )

    conn.commit()
    logger.info("Create DB complete.")
    _close_conn(conn)


def _refine_query_llm(
    text: str,
    conversation_context: list[dict[str, str]] | None = None,
) -> str:
    """
    Use the main LLM (already loaded at app runtime) to rewrite the
    user's conversational query into an embedding-optimized search string.

    Args:
        text (str): raw user query
        conversation_context (list[dict]): recent chat history for intent
    Returns:
        str: refined query string
    """
    from models import main_model

    messages = [{"role": "system", "content": QUERY_REFINE_SYSTEM_PROMPT}]

    # include recent conversation context (last 4 turns max) for intent
    if conversation_context:
        recent = [
            m for m in conversation_context if m["role"] in ("user", "assistant")
        ][-4:]
        for m in recent:
            messages.append({"role": m["role"], "content": m["content"]})

    messages.append({"role": "user", "content": text})

    refined = main_model.generate(messages, max_new_tokens=100)
    logger.info(f"LLM query refinement: '{text}' -> '{refined}'")
    return refined


def _refine_query_keyword(text: str) -> str:
    """
    Lightweight keyword-based query refinement. Extracts meaningful
    terms and expands with common movie-domain synonyms.

    Args:
        text (str): raw user query
    Returns:
        str: refined query string
    """
    # common movie-domain synonym expansions
    SYNONYMS = {
        "scary": "horror frightening suspense",
        "funny": "comedy humor hilarious lighthearted",
        "sad": "drama emotional tearjerker melancholy",
        "romantic": "romance love relationship",
        "action": "action adventure thrilling explosive",
        "dark": "dark gritty noir psychological",
        "kids": "family animation children friendly",
        "old": "classic vintage retro",
        "new": "recent latest contemporary",
        "good": "highly rated acclaimed",
        "best": "top rated acclaimed masterpiece",
        "fun": "entertaining enjoyable lighthearted comedy",
        "intense": "thriller suspense gripping tense",
        "weird": "surreal experimental unconventional avant-garde",
        "lighthearted": "comedy feel-good uplifting heartwarming",
        "war": "war military combat battlefield",
        "space": "sci-fi space science fiction cosmic",
        "mystery": "mystery detective whodunit crime investigation",
    }

    # stop words to filter out
    STOP_WORDS = {
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "can",
        "may",
        "might",
        "shall",
        "must",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "about",
        "like",
        "want",
        "looking",
        "find",
        "show",
        "tell",
        "recommend",
        "suggest",
        "something",
        "movie",
        "movies",
        "film",
        "films",
        "watch",
        "see",
        "seen",
        "watched",
        "that",
        "which",
        "what",
        "this",
        "it",
        "its",
        "but",
        "and",
        "or",
        "not",
        "no",
        "so",
        "if",
        "just",
        "also",
        "really",
        "very",
        "some",
        "any",
        "more",
        "most",
        "much",
        "many",
        "make",
        "makes",
        "made",
        "feel",
        "get",
        "got",
    }

    # extract words, filter stop words
    words = re.findall(r"[a-zA-Z]+", text.lower())
    keywords = [w for w in words if w not in STOP_WORDS and len(w) > 1]

    # expand with synonyms
    expanded = []
    for word in keywords:
        expanded.append(word)
        if word in SYNONYMS:
            expanded.append(SYNONYMS[word])

    refined = " ".join(expanded) if expanded else text
    logger.info(f"Keyword query refinement: '{text}' -> '{refined}'")
    return refined


def refine_query(
    text: str,
    method: str = "llm",
    conversation_context: list[dict[str, str]] | None = None,
) -> str:
    """
    Refine a user query for better embedding retrieval.

    Args:
        text (str): raw user query
        method (str): "llm" for LLM-based, "keyword" for keyword extraction
        conversation_context (list[dict]): chat history (used by LLM method)
    Returns:
        str: refined query
    """
    if method == "llm":
        return _refine_query_llm(text, conversation_context)
    elif method == "keyword":
        return _refine_query_keyword(text)
    else:
        return text


def get_topk_matching_tmdb_ids(
    text: str,
    k: int,
    thresh: float = 0.75,
    refine_method: str | None = "llm",
    conversation_context: list[dict[str, str]] | None = None,
) -> list[int] | None:
    """
    Credit: ChatGPT. This function was 90% written by ChatGPT.
    I adapted the function for this codebase and fixed some potential bugs.

    Get all details of the best matching movies from the database.
    Text is a string that 'describes' the returned movie. The
    embedding model is used to find the best matching movie from
    the database given text.

    Args:
        text (str): user string of movie desc
        k (int): number of matching movies to return
        thresh (float): simility cutoff threshold
        refine_method (str|None): "llm", "keyword", or None to skip refinement
        conversation_context (list[dict]): chat history for query refinement
    Returns:
        tmdb_ids (list[int]): tmdb_ids of movies with highest simility score.
            len(tmdb_ids) = k.
        None: if no movies are found.
    """
    # pre-retrieval query refinement
    if refine_method:
        text = refine_query(
            text,
            method=refine_method,
            conversation_context=conversation_context,
        )

    _, cur = _establish_connection(DB_URL)  # pyright: ignore

    emb_tensor = embedding_model.embed([text])[0]

    try:
        emb_list = emb_tensor.detach().cpu().tolist()
    except AttributeError:
        emb_list = emb_tensor.tolist()

    # Convert to pgvector input format
    emb_str = "[" + ",".join(map(str, emb_list)) + "]"

    sql = """
        WITH query AS (
            SELECT %s::vector AS q
        )
        SELECT m.tmdb_id
        FROM movies m, query
        WHERE 1 - (m.embedding <=> query.q) >= %s
        ORDER BY m.embedding <=> query.q
        LIMIT %s;
    """

    cur.execute(sql, (emb_str, thresh, k))
    rows = cur.fetchall()

    if not rows:
        return None

    return [row[0] for row in rows]


def query_db(query: QueryNoTemplate, params: tuple):
    conn, cur = _establish_connection(DB_URL)  # pyright: ignore

    try:
        cur.execute(query, params)
        results = cur.fetchall()
    except Exception as e:
        logger.error(f"Query raised.\n{e}")
        results = "Error"

    _close_conn(conn)

    return results


def _test_embed():
    warnings.warn(
        "This function will raise. Please dont use it. Im too lazy to implement it."
    )
    conn, cur = _establish_connection(DB_URL)  # pyright: ignore

    query = input("Enter the query: ")

    tmdb_id = query
    print(tmdb_id)

    cur.execute(
        "SELECT title FROM movies WHERE tmdb_id = %s;",
        (tmdb_id,),
    )

    title = cur.fetchone()
    print(f"Title: {title}")

    _close_conn(conn)


if __name__ == "__main__":
    inp = int(input("Enter opts (create_db: 1, test_embed: 2): "))
    if inp == 1:
        batch_size = int(input("Enter enrichment batch size (default 8): ") or "8")
        _create_db(enrich_batch_size=batch_size)
    elif inp == 2:
        _test_embed()
    else:
        raise ValueError(f"Got invalid opt, expected 1/2, got: {inp}")

"""
Launch some n workers to parallelly fetch movies data.

How to use:
    # Assuming that you are running from project root
    python data/fetch_movies.py --top-k _ --workers _ --keep-shards ...

Following are availible opts when running this script:
    --top-k (int, default=500)
    --workers (int, default=6)
    --env (str, default=.env)
    --out (str, default=data/row_movies.csv)
    --shard-dir (str, default=data/shards)
    --delay (float, default=0.07)
    --keep-shards (optional)

Known errors:
    If the progress bar is stuck or the script takes longer than normal to
    complete, then something failed. Just re-run the script. That worked
    for me. At this moment in time, idk what the cause of this bug is, but
    im able to download/fetch the data by re-running so i dont want to break
    my head fixing this thing.

IMPORTANT:
    Run using hotspot, or set manual DNS to cloudflare (1.1.1.1) and/or
    google (8.8.8.8). Jio is blocking TMDB for some reason. Top comment here
    was the solution for me:
    https://www.reddit.com/r/NovaVideoPlayer/comments/1beft2v/information_tmdb_is_not_accessible_in_india/
"""

import argparse
import csv
import logging
import os
import queue
import shutil
import threading
import time
import requests

from dotenv import load_dotenv
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm


BASE_URL = "https://api.themoviedb.org/3"
DISCOVER_URL = f"{BASE_URL}/discover/movie"
DETAIL_URL = f"{BASE_URL}/movie/{{tmdb_id}}"
APPEND = "external_ids,credits,keywords,release_dates"
TIMEOUT = (30, 60)  # (connect_s, read_s)

# Sentinel pushed by each worker when its chunk is done (daemon counts em)
_DONE = object()

FIELDNAMES = [
    "tmdb_id",
    "imdb_id",
    "title",
    "original_title",
    "tagline",
    "overview",
    "release_date",
    "runtime_mins",
    "certification",
    "genres",
    "keywords",
    "spoken_languages",
    "origin_country",
    "collection",
    "director",
    "top_cast",
    "production_companies",
    "budget",
    "revenue",
    "vote_average",
    "vote_count",
    "popularity",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-16s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# Session factory (one per thread)
def build_session(api_key: str) -> requests.Session:
    """Thread (local) Session with bearer auth and retry/backoff."""
    retry = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.headers.update(
        {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        }
    )
    return s


# Pre-flight
def check_connectivity(session: requests.Session) -> None:
    log.info("Checking TMDB connectivity ...")
    try:
        r = session.get(f"{BASE_URL}/configuration", timeout=TIMEOUT)
        r.raise_for_status()
        log.info("TMDB reachable OK")
    except requests.exceptions.ConnectTimeout:
        raise SystemExit("[ERROR] Connection timed out")
    except requests.exceptions.ConnectionError as exc:
        raise SystemExit(f"[ERROR] Cannot reach TMDB: {exc}")
    except requests.exceptions.HTTPError as exc:
        raise SystemExit(f"[ERROR] TMDB rejected the request: {exc}")


# Discovery
def discover_stubs(session: requests.Session, top_k: int) -> list[dict]:
    """
    Fetch lightweight movie stubs via /discover/movie (popularity desc)
    (Single threaded)
    """
    stubs = []
    pages_needed = (top_k + 19) // 20  # each page has 20 movies
    log.info(f"Discovering {top_k} movies ({pages_needed} pages) ...")

    for page in tqdm(
        range(1, pages_needed + 1),
        desc="Discover",
        unit="page(s)",
        leave=False,
    ):
        r = session.get(
            DISCOVER_URL,
            params={
                "sort_by": "popularity.desc",
                "include_adult": False,
                "include_video": False,
                "page": page,
                "vote_count.gte": 50,
                "with_original_language": "en",
            },
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        stubs.extend(data.get("results", []))
        if page >= data.get("total_pages", 1):
            break
        time.sleep(0.05)

    return stubs[:top_k]


# Parsing
# Utils
def _pipe(items: list) -> str:
    return "|".join(str(i) for i in items if i)


def _certification(detail: dict) -> str:
    """
    Return US certification
    Fall back to first available
    """
    fallback = ""
    for country in detail.get("release_dates", {}).get("results", []):
        for rd in country.get("release_dates", []):
            c = rd.get("certification", "").strip()
            if c:
                if country.get("iso_3166_1") == "US":
                    return c
                fallback = fallback or c
    return fallback


def parse_movie(detail: dict) -> dict:
    imdb_id = (
        detail.get("imdb_id") \
        or (detail.get("external_ids") or {}).get("imdb_id", "") \
        or ""
    )
    credits = detail.get("credits", {})
    directors = _pipe(
        [p["name"] for p in credits.get("crew", []) if p.get("job") == "Director"]
    )
    top_cast = _pipe([p["name"] for p in credits.get("cast", [])[:5]])
    return {
        "tmdb_id": detail.get("id", ""),
        "imdb_id": imdb_id,
        "title": detail.get("title", ""),
        "original_title": detail.get("original_title", ""),
        "tagline": detail.get("tagline", ""),
        "overview": (detail.get("overview") or "").replace("\n", " "),
        "release_date": detail.get("release_date", ""),
        "runtime_mins": detail.get("runtime") or "",
        "certification": _certification(detail),
        "genres": _pipe([g["name"] for g in detail.get("genres", [])]),
        "keywords": _pipe(
            [k["name"] for k in detail.get("keywords", {}).get("keywords", [])]
        ),
        "spoken_languages": _pipe(
            [
                lang.get("english_name") or lang.get("name", "")
                for lang in detail.get("spoken_languages", [])
            ]
        ),
        "origin_country": _pipe(detail.get("origin_country", [])),
        "collection": (detail.get("belongs_to_collection") or {}).get("name", ""),
        "director": directors,
        "top_cast": top_cast,
        "production_companies": _pipe(
            [c["name"] for c in detail.get("production_companies", [])]
        ),
        "budget": detail.get("budget", 0),
        "revenue": detail.get("revenue", 0),
        "vote_average": detail.get("vote_average", ""),
        "vote_count": detail.get("vote_count", ""),
        "popularity": detail.get("popularity", ""),
    }


# Worker
def worker(
    worker_id: int,
    chunk: list[dict],
    api_key: str,
    row_queue: queue.Queue,
    shard_path: Path,
    delay: float,
    error_log: list,
    error_lock: threading.Lock,
    progress_bar: tqdm,
) -> None:
    """
    Fetch details for every stub in chunk
        1. Writes each parsed row to shard-path (worker's file).
        2. Then pushes each row into row_queue for the daemon to write
           to the main (master) file.
        3. Finally, pushes _DONE sentinel when finished.
    """
    session = build_session(api_key)

    with shard_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for stub in chunk:
            tmdb_id = stub["id"]
            try:
                r = session.get(
                    DETAIL_URL.format(tmdb_id=tmdb_id),
                    params={"append_to_response": APPEND},
                    timeout=TIMEOUT,
                )
                r.raise_for_status()
                row = parse_movie(r.json())
                writer.writerow(row)
                row_queue.put(row)
            except (
                requests.HTTPError,
                requests.Timeout,
                requests.ConnectionError,
            ) as exc:
                with error_lock:
                    error_log.append((worker_id, tmdb_id, str(exc)))
            finally:
                progress_bar.update(1)
                time.sleep(delay)

    row_queue.put(_DONE)
    log.info(
        f"Worker {worker_id} done  ({len(chunk)} movies, shard: {shard_path.name})"
    )


# Daemon thread
def daemon(
    row_queue: queue.Queue,
    master_path: Path,
    n_workers: int,
) -> None:
    """
    Does stuff...
    Drains the row-queue and append rows to the master file in parallel.
    Umm... thats it.

    Please be careful (future dhruv), this will only exit once it receives
    _DONE from exactly n_workers. So this may run forever if any of the workers
    fail.

    Args:
        row_queue (Queue): row queue
        master_path (Path): path to master file
        n_workers (int): number of workers
    Returns:
        None. Writes to file
    """
    master_path.parent.mkdir(parents=True, exist_ok=True)
    done_count = 0
    written = 0

    with master_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        while done_count < n_workers:
            try:
                # if a daemon fails, this should be able to handle it
                # but its not very explicit, and i dont like that... :(
                item = row_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is _DONE:
                done_count += 1
                log.info(f"Daemon: received DONE {done_count}/{n_workers}")
            else:
                writer.writerow(item)
                written += 1
                # flush every 50 rows so the file is readable (mid-run)
                if written % 50 == 0:
                    f.flush()

        f.flush()

    log.info(f"Daemon: master CSV complete  ({written} rows to {master_path})")


# More helpers
def partition(lst: list, n: int) -> list[list]:
    """Split lst into n "roughly" equal chunks."""
    k, rem = divmod(len(lst), n)
    chunks = []
    start = 0
    for i in range(n):
        size = k + (1 if i < rem else 0)
        chunks.append(lst[start: start + size])
        start += size
    return [c for c in chunks if c]


# ========================================================


def main() -> None:
    # argparse bs
    # todo, write run instructions at top of file to refer to later...
    parser = argparse.ArgumentParser(description="TMDB movie fetcher")
    parser.add_argument(
        "--top-k",
        type=int,
        default=500,
        help="Number of movies to fetch (default: 500)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Number of workers (default: 6)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=".env",
        help="Path to .env file with `TMDB_API_KEY`",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/raw_movies.csv",
        help="Master output CSV (default: data/raw_movies.csv)",
    )
    parser.add_argument(
        "--shard-dir",
        type=str,
        default="data/shards",
        help="Directory for per-worker shard files",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.07,
        help="Per-worker seconds between detail calls (default: 0.07)",
    )
    parser.add_argument(
        "--keep-shards",
        action="store_true",
        help="Keep shard files after run (default: delete them)",
    )
    args = parser.parse_args()

    # load API key
    load_dotenv(args.env)
    api_key = os.getenv("TMDB_API_KEY", "").strip()
    if not api_key:
        raise KeyError(f"[ERROR] TMDB_API_KEY not found in '{args.env}'.")

    # prep stuff
    main_session = build_session(api_key)
    check_connectivity(main_session)

    # get stubs
    stubs = discover_stubs(main_session, args.top_k)
    log.info(f"{len(stubs)} stubs collected")

    # make chunks
    n_workers = min(args.workers, len(stubs))
    chunks = partition(stubs, n_workers)
    log.info(f"Partitioned into {len(chunks)} chunks for {n_workers} workers")

    # path and file stuff
    shard_dir = Path(args.shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    master_path = Path(args.out)

    # shared state
    row_queue = queue.Queue(maxsize=500)
    error_log = []
    error_lock = threading.Lock()

    pbar = tqdm(total=len(stubs), desc="Fetching", unit="movie")
    # start daemon
    daemon_thread = threading.Thread(
        target=daemon,
        args=(row_queue, master_path, len(chunks)),
        name="Daemon",
        daemon=True,  # exits automatically if main thread crashes
    )
    daemon_thread.start()
    log.info(f"Daemon started, master path: {master_path}")

    # start worker thread(s)
    worker_threads = []
    for i, chunk in enumerate(chunks):
        shard_path = shard_dir / f"shard_{i}.csv"
        t = threading.Thread(
            target=worker,
            args=(
                i,
                chunk,
                api_key,
                row_queue,
                shard_path,
                args.delay,
                error_log,
                error_lock,
                pbar,
            ),
            name=f"Worker-{i}",
        )
        t.start()
        worker_threads.append(t)
        log.info(f"Worker {i} started  ({len(chunk)} movies, shard path: {shard_path.name})")

    # wait for workers to finish
    for t in worker_threads:
        t.join()

    pbar.close()

    # wait for daemon to finish
    log.info("All workers done. Waiting for daemon to flush master CSV ...")
    daemon_thread.join()

    # some logging once done
    total_rows = sum(1 for _ in open(master_path, encoding="utf-8")) - 1
    print("\n" + "==" * 50)
    print(f"\nDone. Total of {total_rows} movies saved to {master_path}")
    print(f"Shards: {shard_dir}/shard_0..{len(chunks) - 1}.csv")

    if error_log:
        print(f"\n{len(error_log)} errors:")
        for wid, tmdb_id, msg in error_log[:20]:
            print(f"  [worker {wid}] tmdb_id={tmdb_id}: {msg}")

    # delete shards
    if not args.keep_shards:
        shutil.rmtree(shard_dir)
        log.info(f"Shard directory deleted: {shard_dir}")
    else:
        log.info(f"Shards kept at: {shard_dir}")


if __name__ == "__main__":
    main()

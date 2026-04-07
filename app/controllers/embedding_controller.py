from dataclasses import dataclass

from data.data_engine import get_topk_matching_tmdb_ids, query_db


@dataclass
class EmbeddingControllerConfig:
    k: int
    threshold: float


class EmbeddingController:
    """
    embedding controller. manages information retrieval from the database and
    formatting the output for the chat model.

    once initialized, following functions are useable:
        best_match(prompt: str)
    """

    def __init__(self, config: EmbeddingControllerConfig):
        """nothing to init."""
        self.config = config

    def best_match(
        self,
        prompt: str,
        conversation_context: list[dict[str, str]] | None = None,
        refine_method: str | None = "llm",
    ) -> str:
        """
        find relevant information from the database using prompt. this function
        assumes that the prompt best describes a movie from the database. this
        function will call data_engine.query_db, looking for the best movie's
        title and overview. once that information is found, it will be parsed
        and retruned as a string.

        args:
            prompt (str): user prompt
            conversation_context (list[dict]): chat history for query refinement
            refine_method (str|None): "llm", "keyword", or None
        returns:
            results (str): database results parsed as a string
        """
        print(
            f"[INFO::embedding_controller::best_match] "
            f"received prompt: {prompt}. Fetching best movie details"
        )
        tmdb_ids = get_topk_matching_tmdb_ids(
            prompt,
            k=self.config.k,
            thresh=self.config.threshold,
            refine_method=refine_method,
            conversation_context=conversation_context,
        )
        results = query_db(
            """
            SELECT title, overview, director, top_cast, genres,
                release_date, keywords, spoken_languages
            FROM movies
            WHERE tmdb_id = ANY(%s)
            """,
            (tmdb_ids,),
        )

        output = self._parse(results)
        return output

    def _parse(self, results: list[tuple]) -> str:
        """
        convert db output into a simple string for the chat model.
        """
        output = []
        for row in results:
            (
                title,
                overview,
                director,
                top_cast,
                genres,
                release_date,
                keywords,
                spoken_languages,
            ) = row
            output.append(
                f"Movie title: {title} | "
                f"Movie overview: {overview} | "
                f"Movie director: {director} | "
                f"Movie top_cast: {top_cast} | "
                f"Movie genres: {genres} | "
                f"Movie release_date: {release_date} | "
                f"Movie keywords: {keywords} | "
                f"Movie spoken_languages: {spoken_languages}"
            )

        output = "\n\n".join(output)

        return output

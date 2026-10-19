from typing import Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def cosine_similarity_tool(
        doc_index_1: Optional[int] = None,
        doc_index_2: Optional[int] = None,
        text_1: Optional[str] = None,
        text_2: Optional[str] = None
    ) -> dict:
        """Compute the cosine similarity between two documents specified either by row indices (doc_index_1, doc_index_2) or raw text strings (text_1, text_2).

        Args:
            doc_index_1: Index of the first document in session.feature_matrix.
            doc_index_2: Index of the second document in session.feature_matrix.
            text_1: Raw text string of the first document.
            text_2: Raw text string of the second document.

        Returns:
            A dictionary containing the result ID, inputs, and their cosine similarity score.
        """
        if session.feature_matrix is None:
            return {"error": "feature_matrix is None. Build DTM first."}
        
        X = session.feature_matrix
        n_docs = X.shape[0]

        def resolve_vector(doc_idx, text):
            if doc_idx is not None:
                if not (0 <= doc_idx < n_docs):
                    raise ValueError(f"Document index {doc_idx} out of bounds for matrix with {n_docs} rows.")
                return X[doc_idx]
            elif text is not None:
                vectorizer = session.artifacts.get("count_vectorizer")
                if vectorizer is None:
                    raise ValueError("count_vectorizer not found in session.artifacts. Build DTM first.")
                return vectorizer.transform([text])
            else:
                raise ValueError("Must provide either doc_index or text for the document.")

        try:
            vec1 = resolve_vector(doc_index_1, text_1)
            vec2 = resolve_vector(doc_index_2, text_2)
        except ValueError as e:
            return {"error": str(e)}

        sim = cosine_similarity(vec1, vec2)[0, 0]

        result_id = session.next_result_id("cosine_similarity")
        summary = {
            "result_id": result_id,
            "doc_index_1": doc_index_1,
            "doc_index_2": doc_index_2,
            "text_1": text_1[:30] + "..." if text_1 else None,
            "text_2": text_2[:30] + "..." if text_2 else None,
            "cosine_similarity": float(sim)
        }

        session.store_result(
            "cosine_similarity_tool",
            {"doc_index_1": doc_index_1, "doc_index_2": doc_index_2, "text_1": text_1, "text_2": text_2},
            summary
        )
        return summary

    return [cosine_similarity_tool]

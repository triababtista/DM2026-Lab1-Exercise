from typing import Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

    @tool
    def feature_correlation_matrix_tool() -> dict:
        """Compute the Pearson correlation matrix over the top-20 terms by variance from session.feature_matrix, plot a heatmap, and store the result.

        Returns:
            A dictionary containing the result ID, term names, and correlation matrix values.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "feature_matrix or feature_names is None. Build DTM first."}

        X = session.feature_matrix
        if hasattr(X, "toarray"):
            X_dense = X.toarray()
        else:
            X_dense = np.asarray(X)

        feature_names = np.array(session.feature_names)

        # Compute variance per feature column
        variances = np.var(X_dense, axis=0)
        
        # Sort by variance descending and take top 20 (or all if fewer than 20)
        sorted_indices = np.argsort(variances)[::-1]
        top_k = min(20, len(feature_names))
        top_indices = sorted_indices[:top_k]

        top_names = feature_names[top_indices].tolist()
        sub_matrix = X_dense[:, top_indices]

        # Compute Pearson correlation matrix across columns (rowvar=False)
        corr_matrix = np.corrcoef(sub_matrix, rowvar=False)
        # Handle case where corr_matrix might be scalar if top_k == 1
        if corr_matrix.ndim == 0:
            corr_matrix = np.array([[1.0]])

        # Plotting heatmap with fmt=".2f" for cleaner display
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            xticklabels=top_names,
            yticklabels=top_names,
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            ax=ax
        )
        ax.set_title("Feature-vs-Feature Pearson Correlation Matrix (Top Terms by Variance)")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        session.pending_figure = fig

        result_id = session.next_result_id("feature_correlation")
        summary = {
            "result_id": result_id,
            "term_names": top_names,
            "correlation_matrix": corr_matrix.round(4).tolist()
        }

        session.store_result("feature_correlation_matrix_tool", {}, summary)
        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]

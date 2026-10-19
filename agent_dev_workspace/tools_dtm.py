from langchain_core.tools import tool
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from typing import List, Optional

def make_tools(session):
    
    @tool
    def build_dtm_tool(ngram_range: List[int] = [1, 1], max_features: Optional[int] = None) -> dict:
        """Builds a document-term matrix (DTM) from the working dataset text using CountVectorizer.
        
        Args:
            ngram_range: Lower and upper boundary of the range of n-values for different n-grams to be extracted (passed as a list, e.g. [1, 1]).
            max_features: Optional maximum vocabulary size.
            
        Returns:
            A dictionary containing the status, number of documents, vocabulary size, and sparsity metrics.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"status": "error", "message": "No dataset loaded or 'text' column missing."}
        
        # Convert list parameter to tuple for CountVectorizer
        ngram_tuple = tuple(ngram_range)
        
        vectorizer = CountVectorizer(ngram_range=ngram_tuple, max_features=max_features)
        texts = session.dataframe["text"].fillna("").astype(str).tolist()
        
        feature_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out().tolist()
        
        session.feature_matrix = feature_matrix
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer
        
        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].values)
        
        non_zero = int(feature_matrix.nnz)
        n_rows, n_cols = feature_matrix.shape
        total_elements = n_rows * n_cols
        sparsity_pct = float(100 * (1 - non_zero / total_elements)) if total_elements > 0 else 0.0
        
        result_id = session.next_result_id("build_dtm")
        summary = {
            "result_id": result_id,
            "status": "success",
            "n_documents": n_rows,
            "n_terms": n_cols,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": sparsity_pct
        }
        
        session.store_result(
            tool_name="build_dtm_tool",
            args={"ngram_range": ngram_range, "max_features": max_features},
            summary=summary,
            full_report={"feature_names": feature_names}
        )
        return summary

    @tool
    def term_frequency_tool() -> dict:
        """Aggregates total term frequencies across all documents from the global document-term matrix.
        
        Returns:
            A dictionary containing the status, total unique terms, and top 20 term frequencies.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No document-term matrix built yet. Run build_dtm_tool first."}
        
        # Sum column counts across all rows in feature_matrix
        term_counts = np.array(session.feature_matrix.sum(axis=0)).flatten()
        freq_dict = {term: int(count) for term, count in zip(session.feature_names, term_counts)}
        
        # Sort terms by frequency descending to get top 20 for summary
        sorted_freqs = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
        top_20_dict = dict(sorted_freqs[:20])
        
        result_id = session.next_result_id("term_frequency")
        summary = {
            "result_id": result_id,
            "status": "success",
            "n_unique_terms": len(freq_dict),
            "term_frequencies": top_20_dict
        }
        
        session.store_result(
            tool_name="term_frequency_tool",
            args={},
            summary=summary,
            full_report={"term_frequencies": freq_dict}
        )
        return summary

    @tool
    def dtm_heatmap_tool(n_terms: int = 20, n_documents: int = 20) -> dict:
        """Renders and returns a heatmap of a positional slice of the global document-term matrix.
        
        Args:
            n_terms: Number of terms (columns) to include.
            n_documents: Number of documents (rows) to include.
            
        Returns:
            A dictionary containing the status, sliced matrix, terms, and dimensions.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No document-term matrix built yet. Run build_dtm_tool first."}
        
        # Slice matrix and convert to dense list
        sliced_matrix = session.feature_matrix[0:n_documents, 0:n_terms].toarray()
        sliced_terms = session.feature_names[0:n_terms]
        matrix_list = sliced_matrix.tolist()
        
        # Build heatmap
        fig, ax = plt.subplots(figsize=(max(6, len(sliced_terms) * 0.8), max(4, len(matrix_list) * 0.4)))
        sns.heatmap(sliced_matrix, annot=True, fmt="d", xticklabels=sliced_terms, yticklabels=range(len(matrix_list)), ax=ax, cmap="Blues")
        ax.set_title("Document-Term Matrix Heatmap")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        plt.tight_layout()
        
        session.pending_figure = fig
        
        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": result_id,
            "status": "success",
            "matrix": matrix_list,
            "terms": sliced_terms,
            "n_documents": len(matrix_list),
            "n_terms": len(sliced_terms)
        }
        
        session.store_result(
            tool_name="dtm_heatmap_tool",
            args={"n_terms": n_terms, "n_documents": n_documents},
            summary=summary,
            full_report={"matrix": matrix_list, "terms": sliced_terms}
        )
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]

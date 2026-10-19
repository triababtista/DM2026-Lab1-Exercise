from langchain_core.tools import tool
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from typing import List

def make_tools(session):
    
    @tool
    def build_dtm_tool(ngram_range: List[int] = [1, 1], max_features: int = None) -> dict:
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

    return [build_dtm_tool]

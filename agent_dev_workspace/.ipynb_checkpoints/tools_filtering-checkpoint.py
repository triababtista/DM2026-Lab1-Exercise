import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def variance_filter_tool(threshold: float = 0.0) -> dict:
        """Filters out low-variance terms from the global document-term matrix.

        Args:
            threshold (float): Variance threshold below which terms are removed. Default is 0.0.

        Returns:
            dict: Summary of filtering results including kept/removed counts and terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "Document-term matrix has not been built yet."}

        matrix = session.feature_matrix
        if hasattr(matrix, "toarray"):
            dense_matrix = matrix.toarray()
        else:
            dense_matrix = np.asarray(matrix)

        variances = np.var(dense_matrix, axis=0)
        feature_names = session.feature_names

        # Build full report DataFrame with term name as first column and variance as second column
        report_df = pd.DataFrame({
            "term": feature_names,
            "variance": variances
        })
        report_df = report_df.sort_values(by="variance", ascending=False).reset_index(drop=True)

        # Filter terms based on threshold
        kept_mask = report_df["variance"] >= threshold
        kept_df = report_df[kept_mask]
        removed_df = report_df[~kept_mask]

        result_id = session.next_result_id("variance_filter")

        session.store_result(
            tool_name="variance_filter_tool",
            args={"threshold": threshold},
            summary={
                "result_id": result_id,
                "n_initial_terms": len(feature_names),
                "n_kept_terms": len(kept_df),
                "n_removed_terms": len(removed_df),
                "kept_terms": kept_df["term"].tolist(),
                "removed_terms": removed_df["term"].tolist()
            },
            full_report=report_df
        )

        return {
            "result_id": result_id,
            "n_initial_terms": len(feature_names),
            "n_kept_terms": len(kept_df),
            "n_removed_terms": len(removed_df),
            "kept_terms": kept_df["term"].tolist(),
            "removed_terms": removed_df["term"].tolist()
        }

    @tool
    def pearson_filter_tool(target_class: str) -> dict:
        """Computes Pearson correlation between each term's count and a target class label.

        Args:
            target_class (str): The target category name (one-vs-rest binary indicator).

        Returns:
            dict: Summary of Pearson correlation results including top correlated terms.
        """
        if session.feature_matrix is None or session.feature_names is None or session.dataframe is None:
            return {"error": "Document-term matrix or dataset has not been built/loaded yet."}

        matrix = session.feature_matrix
        if hasattr(matrix, "toarray"):
            dense_matrix = matrix.toarray()
        else:
            dense_matrix = np.asarray(matrix)

        feature_names = session.feature_names
        
        # Build binary target vector for one-vs-rest against target_class
        y = (session.dataframe["category_name"] == target_class).astype(float).values

        pearson_r_values = []
        y_std = np.std(y)

        for col_idx in range(dense_matrix.shape[1]):
            x = dense_matrix[:, col_idx]
            x_std = np.std(x)
            
            # If either x or y has zero variance, correlation is undefined -> set to 0.0
            if x_std == 0 or y_std == 0:
                pearson_r_values.append(0.0)
            else:
                corr = np.corrcoef(x, y)[0, 1]
                if np.isnan(corr):
                    pearson_r_values.append(0.0)
                else:
                    pearson_r_values.append(corr)

        # Build full report DataFrame with term name as first column and pearson_r as second column
        report_df = pd.DataFrame({
            "term": feature_names,
            "pearson_r": pearson_r_values
        })
        report_df = report_df.sort_values(by="pearson_r", ascending=False).reset_index(drop=True)

        result_id = session.next_result_id("pearson_filter")

        session.store_result(
            tool_name="pearson_filter_tool",
            args={"target_class": target_class},
            summary={
                "result_id": result_id,
                "target_class": target_class,
                "correlations": dict(zip(report_df["term"], report_df["pearson_r"]))
            },
            full_report=report_df
        )

        return {
            "result_id": result_id,
            "target_class": target_class,
            "correlations": dict(zip(report_df["term"], report_df["pearson_r"]))
        }

    @tool
    def spearman_filter_tool(target_class: str) -> dict:
        """Computes Spearman rank correlation between each term's count and a target class label.

        Args:
            target_class (str): The target category name (one-vs-rest binary indicator).

        Returns:
            dict: Summary of Spearman correlation results including top correlated terms.
        """
        if session.feature_matrix is None or session.feature_names is None or session.dataframe is None:
            return {"error": "Document-term matrix or dataset has not been built/loaded yet."}

        matrix = session.feature_matrix
        if hasattr(matrix, "toarray"):
            dense_matrix = matrix.toarray()
        else:
            dense_matrix = np.asarray(matrix)

        feature_names = session.feature_names
        
        # Build binary target vector for one-vs-rest against target_class
        y = (session.dataframe["category_name"] == target_class).astype(float).values

        spearman_r_values = []
        y_std = np.std(y)

        for col_idx in range(dense_matrix.shape[1]):
            x = dense_matrix[:, col_idx]
            x_std = np.std(x)
            
            # If either x or y has zero variance, correlation is undefined -> set to 0.0
            if x_std == 0 or y_std == 0:
                spearman_r_values.append(0.0)
            else:
                res = spearmanr(x, y)
                corr = res.correlation
                if np.isnan(corr):
                    spearman_r_values.append(0.0)
                else:
                    spearman_r_values.append(corr)

        # Build full report DataFrame with term name as first column and spearman_r as second column
        report_df = pd.DataFrame({
            "term": feature_names,
            "spearman_r": spearman_r_values
        })
        report_df = report_df.sort_values(by="spearman_r", ascending=False).reset_index(drop=True)

        result_id = session.next_result_id("spearman_filter")

        session.store_result(
            tool_name="spearman_filter_tool",
            args={"target_class": target_class},
            summary={
                "result_id": result_id,
                "target_class": target_class,
                "correlations": dict(zip(report_df["term"], report_df["spearman_r"]))
            },
            full_report=report_df
        )

        return {
            "result_id": result_id,
            "target_class": target_class,
            "correlations": dict(zip(report_df["term"], report_df["spearman_r"]))
        }

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]

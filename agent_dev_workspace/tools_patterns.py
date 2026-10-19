import os
import tempfile
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool
from PAMI.extras.convert.DF2DB import DF2DB
from PAMI.frequentPattern.basic import FPGrowth as fpg_alg
from PAMI.frequentPattern.topk import FAE as topk_alg
from PAMI.frequentPattern.maximal import MaxFPGrowth as max_alg


def make_tools(session):
    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: str = "tfidf",
        algorithm: str = "fpgrowth",
        min_sup: int = 2,
        k: int = 1
    ) -> dict:
        """Mines frequent patterns from a specific category\'s documents using PAMI algorithms.

        Args:
            category_name (str): Which category to filter and mine.
            filtering_method (str): Vocabulary filter ("variance", "tfidf", or "term_frequency"). Default is "tfidf".
            algorithm (str): Mining algorithm ("fpgrowth", "topk", or "maxfpgrowth"). Default is "fpgrowth".
            min_sup (int, optional): Minimum support threshold for fpgrowth / maxfpgrowth. Default is 2.
            k (int, optional): Number of top patterns for topk. Default is 1.

        Returns:
            dict: Summary of mined patterns or filtering notes.
        """
        if session.dataframe is None:
            return {"error": "Dataset has not been loaded yet."}

        # 1. Filter session.dataframe to category_name
        df_cat = session.dataframe[session.dataframe["category_name"] == category_name]
        if df_cat.empty:
            return {"patterns": [], "note": f"No documents found for category {category_name}"}

        texts = df_cat["text"].fillna("").tolist()

        # Build fresh, category-local document-term matrix with stop_words="english"
        vectorizer = CountVectorizer(stop_words="english", strip_accents="unicode")
        X = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        if len(feature_names) == 0 or X.shape[0] == 0:
            return {"patterns": [], "note": "no terms survived stopword removal or empty text"}

        # 2. Apply filtering method
        dense_X = X.toarray()
        keep_indices = []

        if filtering_method == "variance":
            variances = pd.DataFrame(dense_X).var(axis=0).values
            if len(variances) > 0:
                p5, p95 = np.percentile(variances, [5, 95]) if len(variances) >= 2 else (0, 0)
                keep_indices = [i for i, v in enumerate(variances) if p5 <= v <= p95]
        elif filtering_method == "term_frequency":
            col_sums = dense_X.sum(axis=0)
            if len(col_sums) > 0:
                p5, p95 = np.percentile(col_sums, [5, 95]) if len(col_sums) >= 2 else (0, 0)
                keep_indices = [i for i, s in enumerate(col_sums) if p5 <= s <= p95]
        elif filtering_method == "tfidf":
            transformer = TfidfTransformer()
            tfidf_matrix = transformer.fit_transform(X).toarray()
            mean_tfidf = tfidf_matrix.mean(axis=0)
            if len(mean_tfidf) > 0:
                p20 = np.percentile(mean_tfidf, 20) if len(mean_tfidf) >= 2 else 0
                keep_indices = [i for i, m in enumerate(mean_tfidf) if m >= p20]
        else:
            return {"error": f"Unknown filtering_method: {filtering_method}"}

        # 3. Handle zero surviving terms
        if not keep_indices:
            return {"patterns": [], "note": "no terms survived filtering"}

        filtered_dense = dense_X[:, keep_indices]
        filtered_names = feature_names[keep_indices]

        # Convert to DataFrame for DF2DB conversion
        filtered_df = pd.DataFrame(filtered_dense, columns=filtered_names)
        # Drop rows that have all zeros after filtering (empty transactions)
        filtered_df = filtered_df.loc[(filtered_df != 0).any(axis=1)]

        if filtered_df.empty:
            return {"patterns": [], "note": "no terms survived filtering"}

        # 4. Convert and Mine using PAMI (using encoding="utf-8" to prevent UnicodeDecodeError)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, suffix=".csv") as tmp:
            tmp_path = tmp.name

        try:
            db_converter = DF2DB(filtered_df)
            db_converter.convert2TransactionalDatabase(tmp_path, ">=", 1)

            patterns_list = []

            if algorithm == "fpgrowth":
                obj = fpg_alg.FPGrowth(tmp_path, minSup=min_sup, sep="\t")
            elif algorithm == "topk":
                obj = topk_alg.FAE(tmp_path, k=k, sep="\t")
            elif algorithm == "maxfpgrowth":
                obj = max_alg.MaxFPGrowth(tmp_path, minSup=min_sup, sep="\t")
            else:
                return {"error": f"Unknown algorithm: {algorithm}"}

            obj.mine()
            df_patterns = obj.getPatternsAsDataFrame()

            if df_patterns is not None and not df_patterns.empty:
                pattern_col = next((c for c in ["pattern", "patterns", "itemset"] if c in df_patterns.columns), df_patterns.columns[0])
                support_col = next((c for c in ["support", "supp", "count"] if c in df_patterns.columns), df_patterns.columns[1] if len(df_patterns.columns) > 1 else df_patterns.columns[0])
                for _, row in df_patterns.iterrows():
                    patterns_list.append({
                        "pattern": str(row[pattern_col]),
                        "support": int(row[support_col])
                    })

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = patterns_list

        result_id = session.next_result_id("mine_patterns")
        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "n_patterns": len(patterns_list),
            "patterns": patterns_list[:10]  # Truncate to top 10 patterns to avoid LLM context bloat
        }

        session.store_result(
            tool_name="mine_patterns_tool",
            args={
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "min_sup": min_sup,
                "k": k
            },
            summary=summary,
            full_report=pd.DataFrame(patterns_list) if patterns_list else None
        )

        return summary

    return [mine_patterns_tool]

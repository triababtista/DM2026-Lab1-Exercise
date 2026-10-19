from langchain_core.tools import tool
import pandas as pd
import nltk
import os
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns

def make_tools(session):
    
    @tool
    def tokenize_tool() -> dict:
        """Tokenizes each document's text in the working dataset into unigrams.
        
        Saves the resulting token lists in session.dataframe['unigrams'] and
        session.artifacts['tokens'], and records the result in the session results store.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"status": "error", "message": "No dataset loaded or 'text' column missing."}
        
        tokens_list = []
        for text in session.dataframe["text"]:
            if pd.isna(text) or not str(text).strip():
                tokens_list.append([])
            else:
                tokens_list.append(nltk.word_tokenize(str(text).lower()))
                
        session.dataframe["unigrams"] = tokens_list
        session.artifacts["tokens"] = tokens_list
        
        result_id = session.next_result_id("tokenize")
        summary = {
            "result_id": result_id,
            "status": "success",
            "n_documents": len(tokens_list)
        }
        session.store_result(
            tool_name="tokenize_tool",
            args={},
            summary=summary,
            full_report={"tokens": tokens_list}
        )
        return summary

    @tool
    def list_files_tool(subdirectory: str = ".") -> dict:
        """Lists files and folders in the specified repository subdirectory.
        
        Args:
            subdirectory: The relative path of the directory to list files and folders from.
            
        Returns:
            A dictionary containing lists of directories and files found in the subdirectory.
        """
        target_path = Path(subdirectory)
        if not target_path.exists() or not target_path.is_dir():
            return {"status": "error", "message": f"Directory not found: {subdirectory}"}
        
        directories = sorted([p.name for p in target_path.iterdir() if p.is_dir() and not p.name.startswith('.')])
        files = sorted([p.name for p in target_path.iterdir() if p.is_file() and not p.name.startswith('.')])
        
        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "status": "success",
            "subdirectory": subdirectory,
            "directories": directories,
            "files": files
        }
        session.store_result(
            tool_name="list_files_tool",
            args={"subdirectory": subdirectory},
            summary=summary,
            full_report={"directories": directories, "files": files}
        )
        return summary

    @tool
    def inspect_data_tool(n: int = 5) -> dict:
        """Peeks at a specified number of rows from the working DataFrame.
        
        Args:
            n: The number of rows to inspect from the top of the working DataFrame.
            
        Returns:
            A dictionary containing the status, total rows available, and a list of sampled row dictionaries.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset currently loaded in session."}
        
        total_rows = len(session.dataframe)
        preview_df = session.dataframe.head(n)
        records = preview_df.to_dict(orient="records")
        
        result_id = session.next_result_id("inspect_data")
        summary = {
            "result_id": result_id,
            "status": "success",
            "total_rows": total_rows,
            "n_inspected": len(records),
            "preview": records
        }
        session.store_result(
            tool_name="inspect_data_tool",
            args={"n": n},
            summary=summary,
            full_report={"preview": records}
        )
        return summary

    @tool
    def check_missing_tool() -> dict:
        """Checks the working dataset for missing or empty text values.
        
        Returns:
            A dictionary containing the status, total missing count, and indices of missing rows.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"status": "error", "message": "No dataset loaded or 'text' column missing."}
        
        df = session.dataframe
        missing_mask = df["text"].isna() | df["text"].astype(str).str.strip().eq("")
        missing_indices = df[missing_mask].index.tolist()
        n_missing = len(missing_indices)
        
        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "status": "success",
            "n_missing": n_missing,
            "missing_indices": missing_indices
        }
        session.store_result(
            tool_name="check_missing_tool",
            args={},
            summary=summary,
            full_report={"missing_indices": missing_indices}
        )
        return summary

    @tool
    def check_duplicates_tool(drop: bool = False) -> dict:
        """Checks for duplicate documents in the working dataset and optionally drops them.
        
        Args:
            drop: If True, drops duplicate documents in place using keep=False.
            
        Returns:
            A dictionary containing the status, duplicate count, and whether they were dropped.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset currently loaded in session."}
        
        df = session.dataframe
        duplicate_mask = df.duplicated()
        n_duplicates = int(duplicate_mask.sum())
        
        dropped = False
        if drop and n_duplicates > 0:
            df.drop_duplicates(keep=False, inplace=True)
            dropped = True

        result_id = session.next_result_id("check_duplicates")
        summary = {
            "result_id": result_id,
            "status": "success",
            "n_duplicates": n_duplicates,
            "dropped": dropped,
            "current_n_rows": len(df)
        }
        session.store_result(
            tool_name="check_duplicates_tool",
            args={"drop": drop},
            summary=summary,
            full_report={"duplicate_mask": duplicate_mask.tolist()}
        )
        return summary

    @tool
    def sample_data_tool(n: int = 5, random_state: Optional[int] = None) -> dict:
        """Subsamples a specified number of rows from the working DataFrame.
        
        Args:
            n: Number of rows to sample.
            random_state: Optional seed for reproducible sampling.
            
        Returns:
            A dictionary containing the status, number of sampled rows, and top sampled records.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset currently loaded in session."}
        
        df = session.dataframe
        sampled_df = df.sample(n=min(n, len(df)), random_state=random_state)
        records = sampled_df.to_dict(orient="records")
        
        result_id = session.next_result_id("sample_data")
        summary = {
            "result_id": result_id,
            "status": "success",
            "n_sampled": len(records),
            "sample": records[:10]
        }
        session.store_result(
            tool_name="sample_data_tool",
            args={"n": n, "random_state": random_state},
            summary=summary,
            full_report={"sample": records}
        )
        return summary

    @tool
    def describe_data_tool() -> dict:
        """Computes descriptive statistics on document text length and plots a box plot by category.
        
        Returns:
            A dictionary containing overall and per-category text length descriptive statistics.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"status": "error", "message": "No dataset loaded or 'text' column missing."}
        
        df = session.dataframe.copy()
        df["text_length"] = df["text"].fillna("").apply(len)
        
        overall_desc = df["text_length"].describe().to_dict()
        
        category_desc = {}
        if "category_name" in df.columns:
            for cat, group in df.groupby("category_name"):
                category_desc[str(cat)] = group["text_length"].describe().to_dict()
        
        # Build box plot and set on session.pending_figure
        fig, ax = plt.subplots(figsize=(8, 6))
        if "category_name" in df.columns:
            sns.boxplot(x="category_name", y="text_length", data=df, ax=ax)
        else:
            sns.boxplot(y="text_length", data=df, ax=ax)
        ax.set_title("Document Text Length by Category")
        ax.set_xlabel("Category" if "category_name" in df.columns else "")
        ax.set_ylabel("Text Length (Characters)")
        plt.tight_layout()
        
        session.pending_figure = fig
        
        result_id = session.next_result_id("describe_data")
        summary = {
            "result_id": result_id,
            "status": "success",
            "overall": overall_desc,
            "per_category": category_desc
        }
        session.store_result(
            tool_name="describe_data_tool",
            args={},
            summary=summary,
            full_report={"overall": overall_desc, "per_category": category_desc}
        )
        return summary

    return [tokenize_tool, list_files_tool, inspect_data_tool, check_missing_tool, check_duplicates_tool, sample_data_tool, describe_data_tool]

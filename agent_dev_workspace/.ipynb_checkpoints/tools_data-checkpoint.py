from langchain_core.tools import tool
import pandas as pd
import nltk

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

    return [tokenize_tool]

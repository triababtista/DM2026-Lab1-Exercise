import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools

def test_tokenize_tool():
    session = SessionState()
    
    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Get student-built tokenize_tool
    student_tools = make_tools(session)
    tokenize_tool = next(t for t in student_tools if t.name == "tokenize_tool")
    
    # Invoke tokenize_tool
    res = tokenize_tool.invoke({})
    assert res["status"] == "success"
    assert res["n_documents"] == 8
    
    # Check against known answer #10 in TEST_FIXTURE.md
    expected_tokens = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        []
    ]
    
    assert session.dataframe["unigrams"].tolist() == expected_tokens
    assert session.artifacts["tokens"] == expected_tokens

def test_list_files_tool():
    session = SessionState()
    tools = make_tools(session)
    list_files_tool = next(t for t in tools if t.name == "list_files_tool")
    
    # Testing against known answer #11 in TEST_FIXTURE.md
    result = list_files_tool.invoke({"subdirectory": "newdataset"})
    
    assert result["status"] == "success"
    assert result["directories"] == []
    assert result["files"] == ["Reddit-stock-sentiment.csv"]

def test_inspect_data_tool_head():
    session = SessionState()
    
    # Load dataset
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    tools = make_tools(session)
    inspect_data = next(t for t in tools if t.name == "inspect_data_tool")
    
    # Inspect first 3 rows
    result = inspect_data.invoke({"n": 3})
    assert result["status"] == "success"
    assert result["total_rows"] == 8
    assert result["n_inspected"] == 3
    assert len(result["preview"]) == 3
    assert result["preview"][0]["text"] == "always alpha alpha alpha beta"
    assert result["preview"][2]["text"] == "always alpha beta"

def test_inspect_data_tool_no_dataset():
    session = SessionState()
    
    tools = make_tools(session)
    inspect_data = next(t for t in tools if t.name == "inspect_data_tool")
    
    # Invoke without loading dataset
    result = inspect_data.invoke({"n": 3})
    assert result["status"] == "error"
    assert "No dataset" in result["message"]

def test_check_missing_tool():
    session = SessionState()
    
    # Load dataset (sample_fixture.csv)
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    tools = make_tools(session)
    check_missing = next(t for t in tools if t.name == "check_missing_tool")
    
    # Test against Known Answer #2 in TEST_FIXTURE.md (Exactly 1 missing/empty row)
    result = check_missing.invoke({})
    assert result["status"] == "success"
    assert result["n_missing"] == 1
    assert 7 in result["missing_indices"]

def test_check_duplicates_tool():
    session = SessionState()
    
    # Load dataset (sample_fixture.csv)
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    tools = make_tools(session)
    check_duplicates = next(t for t in tools if t.name == "check_duplicates_tool")
    
    # Test checking duplicates without dropping (Known Answer #3)
    res_check = check_duplicates.invoke({"drop": False})
    assert res_check["status"] == "success"
    assert res_check["n_duplicates"] == 1
    assert res_check["dropped"] is False
    assert len(session.dataframe) == 8
    
    # Test dropping duplicates with keep=False (leaving 6 rows as per Known Answer #3)
    res_drop = check_duplicates.invoke({"drop": True})
    assert res_drop["status"] == "success"
    assert res_drop["dropped"] is True
    assert len(session.dataframe) == 6

def test_sample_data_tool():
    session = SessionState()
    
    # Load dataset (sample_fixture.csv)
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    tools = make_tools(session)
    sample_data = next(t for t in tools if t.name == "sample_data_tool")
    
    # Test against Known Answer #13 in TEST_FIXTURE.md (n=4, random_state=42 -> rows [1, 5, 0, 7])
    result = sample_data.invoke({"n": 4, "random_state": 42})
    assert result["status"] == "success"
    assert result["n_sampled"] == 4
    
    # Retrieve full report to verify sampled rows
    full_report = session.results_store[result["result_id"]]["full_report"]
    sample_records = full_report["sample"]
    
    # Recreate the expected sample using pandas directly on the same fixture to verify row index indices
    import pandas as pd
    df = pd.read_csv("agent_dev/sample_fixture.csv")
    expected_indices = df.sample(n=4, random_state=42).index.tolist()
    sampled_indices = [row.get("index") for row in sample_records if "index" in row] if sample_records else None
    
    # Alternatively verify by matching the text content of the sampled rows
    expected_texts = df.loc[expected_indices, "text"].fillna("").tolist()
    actual_texts = [row["text"] for row in sample_records]
    assert actual_texts == expected_texts

def test_describe_data_tool():
    session = SessionState()
    
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    tools = make_tools(session)
    describe_tool = next(t for t in tools if t.name == "describe_data_tool")
    
    result = describe_tool.invoke({})
    assert result["status"] == "success"
    
    # Check text lengths per document against Known Answer #18
    expected_lengths = [29, 23, 17, 30, 24, 18, 18, 0]
    actual_lengths = session.dataframe["text"].fillna("").apply(len).tolist()
    assert actual_lengths == expected_lengths
    
    overall = result["overall"]
    assert overall["count"] == 8.0
    assert overall["mean"] == pytest.approx(19.875, abs=1e-3)
    assert overall["std"] == pytest.approx(9.4330, abs=1e-3)
    assert overall["min"] == 0.0
    assert overall["25%"] == pytest.approx(17.75, abs=1e-3)
    assert overall["50%"] == pytest.approx(20.5, abs=1e-3)
    assert overall["75%"] == pytest.approx(25.25, abs=1e-3)
    assert overall["max"] == 30.0
    
    per_cat = result["per_category"]
    catA = per_cat["catA"]
    assert catA["count"] == 4.0
    assert catA["mean"] == pytest.approx(17.25, abs=1e-3)
    assert catA["std"] == pytest.approx(12.5, abs=1e-3)
    assert catA["min"] == 0.0
    assert catA["25%"] == pytest.approx(12.75, abs=1e-3)
    assert catA["50%"] == pytest.approx(20.0, abs=1e-3)
    assert catA["75%"] == pytest.approx(24.5, abs=1e-3)
    assert catA["max"] == 29.0
    
    catB = per_cat["catB"]
    assert catB["count"] == 4.0
    assert catB["mean"] == pytest.approx(22.5, abs=1e-3)
    assert catB["std"] == pytest.approx(5.7446, abs=1e-3)
    assert catB["min"] == 18.0
    assert catB["25%"] == pytest.approx(18.0, abs=1e-3)
    assert catB["50%"] == pytest.approx(21.0, abs=1e-3)
    assert catB["75%"] == pytest.approx(25.5, abs=1e-3)
    assert catB["max"] == 30.0
    
    assert session.pending_figure is not None

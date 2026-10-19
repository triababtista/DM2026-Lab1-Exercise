import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools

def test_build_dtm_tool():
    session = SessionState()
    
    # Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Get student-built build_dtm_tool
    dtm_tools = make_tools(session)
    build_dtm = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    
    # Invoke build_dtm_tool
    result = build_dtm.invoke({"ngram_range": [1, 1], "max_features": None})
    
    assert result["status"] == "success"
    assert result["n_documents"] == 8
    
    # Check against Known Answer #4 in TEST_FIXTURE.md
    # Vocabulary: ['alpha', 'always', 'beta', 'delta', 'gamma'] (5 terms)
    expected_vocab = ["alpha", "always", "beta", "delta", "gamma"]
    assert sorted(session.feature_names) == expected_vocab
    assert result["n_terms"] == 5
    
    # Sparsity: 21 non-zero cells out of 40 total, sparsity_pct = 47.5
    assert result["non_zero"] == 21
    assert result["total_elements"] == 40
    assert result["sparsity_pct"] == pytest.approx(47.5, abs=1e-3)
    
    # Check session artifacts and labels
    assert "count_vectorizer" in session.artifacts
    assert session.feature_matrix.shape == (8, 5)
    assert session.labels is not None
    assert len(session.categories) == 2

def test_term_frequency_tool():
    session = SessionState()
    
    # Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    dtm_tools = make_tools(session)
    build_dtm = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    term_freq_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")
    
    # Build DTM first
    build_dtm.invoke({"ngram_range": [1, 1], "max_features": None})
    
    # Invoke term_frequency_tool
    result = term_freq_tool.invoke({})
    
    assert result["status"] == "success"
    
    # Check against Known Answer #14 in TEST_FIXTURE.md:
    # alpha: 6, always: 7, beta: 3, delta: 4, gamma: 7
    expected_frequencies = {
        "alpha": 6,
        "always": 7,
        "beta": 3,
        "delta": 4,
        "gamma": 7
    }
    
    assert result["term_frequencies"] == expected_frequencies

def test_dtm_heatmap_tool():
    session = SessionState()
    
    # Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    dtm_tools = make_tools(session)
    build_dtm = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_heatmap = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")
    
    # Build DTM first
    build_dtm.invoke({"ngram_range": [1, 1], "max_features": None})
    
    # Test default slice (Known Answer #21 full matrix)
    result_default = dtm_heatmap.invoke({})
    assert result_default["status"] == "success"
    assert result_default["n_documents"] == 8
    assert result_default["n_terms"] == 5
    assert result_default["terms"] == ["alpha", "always", "beta", "delta", "gamma"]
    
    expected_full_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0]
    ]
    assert result_default["matrix"] == expected_full_matrix
    assert session.pending_figure is not None
    
    # Test custom slice (Known Answer #21 3x2 corner)
    result_custom = dtm_heatmap.invoke({"n_terms": 2, "n_documents": 3})
    assert result_custom["status"] == "success"
    assert result_custom["n_documents"] == 3
    assert result_custom["n_terms"] == 2
    assert result_custom["terms"] == ["alpha", "always"]
    
    expected_corner_matrix = [
        [3, 1],
        [2, 1],
        [1, 1]
    ]
    assert result_custom["matrix"] == expected_corner_matrix
    assert session.pending_figure is not None

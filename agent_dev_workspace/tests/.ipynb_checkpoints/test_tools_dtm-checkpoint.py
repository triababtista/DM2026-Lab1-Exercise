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

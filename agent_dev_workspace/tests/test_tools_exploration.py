import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools

def setup_session_with_dtm():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    return session

def test_cosine_similarity_known_answers():
    session = setup_session_with_dtm()
    exploration_tools = make_exploration_tools(session)
    cosine_sim_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")
    
    # 0 vs 1 (both catA, near-identical text): 0.9847
    res_0_1 = cosine_sim_tool.invoke({"doc_index_1": 0, "doc_index_2": 1})
    assert res_0_1["cosine_similarity"] == pytest.approx(0.9847, abs=1e-3)
    
    # 0 vs 3 (catA vs catB, share only "always"): 0.0909
    res_0_3 = cosine_sim_tool.invoke({"doc_index_1": 0, "doc_index_2": 3})
    assert res_0_3["cosine_similarity"] == pytest.approx(0.0909, abs=1e-3)
    
    # 3 vs 4 (both catB, near-identical text): 0.9847
    res_3_4 = cosine_sim_tool.invoke({"doc_index_1": 3, "doc_index_2": 4})
    assert res_3_4["cosine_similarity"] == pytest.approx(0.9847, abs=1e-3)
    
    # 0 vs 7 (catA vs the empty-text row): 0.0
    res_0_7 = cosine_sim_tool.invoke({"doc_index_1": 0, "doc_index_2": 7})
    assert res_0_7["cosine_similarity"] == pytest.approx(0.0, abs=1e-3)

def test_feature_correlation_matrix_known_answer():
    session = setup_session_with_dtm()
    exploration_tools = make_exploration_tools(session)
    corr_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")
    
    res = corr_tool.invoke({})
    
    assert "result_id" in res
    term_names = res["term_names"]
    corr_matrix = np.array(res["correlation_matrix"])
    
    # By variance descending on sample_fixture.csv: alpha, gamma, delta, beta, always
    expected_terms = ["alpha", "gamma", "delta", "beta", "always"]
    assert term_names == expected_terms
    
    # Check shape (5x5)
    assert corr_matrix.shape == (5, 5)
    
    # Diagonal should be 1.0
    for i in range(5):
        assert corr_matrix[i, i] == pytest.approx(1.0, abs=1e-4)
        
    # Check specific known correlations from TEST_FIXTURE.md #19:
    # alpha / beta correlation (0.8885)
    alpha_idx = term_names.index("alpha")
    beta_idx = term_names.index("beta")
    assert corr_matrix[alpha_idx, beta_idx] == pytest.approx(0.8885, abs=1e-3)
    
    # gamma / delta correlation (0.8307)
    gamma_idx = term_names.index("gamma")
    delta_idx = term_names.index("delta")
    assert corr_matrix[gamma_idx, delta_idx] == pytest.approx(0.8307, abs=1e-3)
    
    # Verify session.pending_figure is set
    assert session.pending_figure is not None

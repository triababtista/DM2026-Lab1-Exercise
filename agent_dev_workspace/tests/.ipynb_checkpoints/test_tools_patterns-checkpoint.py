import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools as make_pattern_tools


def test_mine_patterns_fpgrowth_tfidf():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    pattern_tools = make_pattern_tools(session)
    mine_patterns_tool = next(t for t in pattern_tools if t.name == "mine_patterns_tool")

    # Load dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Test catA with tfidf and fpgrowth (Known Answer #17)
    res_cata = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 3
    })
    assert len(res_cata["patterns"]) == 1
    assert res_cata["patterns"][0]["support"] == 3
    # pattern string format from PAMI can vary slightly in formatting (e.g. "alpha" or "{alpha}"), check containment
    assert "alpha" in res_cata["patterns"][0]["pattern"]

    # Test catB with tfidf and fpgrowth (Known Answer #17)
    res_catb = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4
    })
    assert len(res_catb["patterns"]) == 1
    assert res_catb["patterns"][0]["support"] == 4
    assert "gamma" in res_catb["patterns"][0]["pattern"]


def test_mine_patterns_zero_terms_survived():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    pattern_tools = make_pattern_tools(session)
    mine_patterns_tool = next(t for t in pattern_tools if t.name == "mine_patterns_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Variance filtering keeps zero terms on this fixture (Known Answer #17)
    res = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "variance",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res["patterns"] == []


def test_mine_patterns_topk_and_maxfpgrowth():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    pattern_tools = make_pattern_tools(session)
    mine_patterns_tool = next(t for t in pattern_tools if t.name == "mine_patterns_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Test topk algorithm (Known Answer #20)
    res_topk = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert len(res_topk["patterns"]) == 1
    assert "alpha" in res_topk["patterns"][0]["pattern"]
    assert res_topk["patterns"][0]["support"] == 3

    # Test maxfpgrowth algorithm returns zero patterns (Known Answer #20)
    res_max = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert len(res_max["patterns"]) == 0

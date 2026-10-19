import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def test_variance_filter_tool():
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load DTM and filtering tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    # 1. Load dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 2. Build DTM
    build_dtm_tool.invoke({})

    # 3. Test variance filter with threshold = 0.15 (Known Answer #5)
    res = variance_filter_tool.invoke({"threshold": 0.15})

    assert res["n_kept_terms"] == 4
    assert res["n_removed_terms"] == 1

    # Check full report against known answer #5
    full_report = session.results_store[res["result_id"]]["full_report"]
    variance_dict = dict(zip(full_report["term"], full_report["variance"]))

    assert "always" not in full_report[full_report["variance"] >= 0.15]["term"].values
    assert variance_dict["alpha"] == pytest.approx(1.1875, rel=1e-3)
    assert variance_dict["always"] == pytest.approx(0.109375, rel=1e-3)
    assert variance_dict["beta"] == pytest.approx(0.234375, rel=1e-3)
    assert variance_dict["delta"] == pytest.approx(0.25, rel=1e-3)
    assert variance_dict["gamma"] == pytest.approx(1.109375, rel=1e-3)


def test_pearson_filter_tool():
    # Fresh session for this test function to prevent state carryover
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    build_dtm_tool.invoke({})

    # Test Pearson correlation against "catB" (Known Answer #6)
    res = pearson_filter_tool.invoke({"target_class": "catB"})
    full_report = session.results_store[res["result_id"]]["full_report"]
    correlations = dict(zip(full_report["term"], full_report["pearson_r"]))

    # Assertions against Known Answer #6 Pearson r values
    assert correlations["alpha"] == pytest.approx(-0.6882, abs=1e-3)
    assert correlations["always"] == pytest.approx(0.3780, abs=1e-3)
    assert correlations["beta"] == pytest.approx(-0.7746, abs=1e-3)
    assert correlations["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert correlations["gamma"] == pytest.approx(0.8307, abs=1e-3)


def test_spearman_filter_tool():
    # Fresh session for this test function to prevent state carryover
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    build_dtm_tool.invoke({})

    # Test Spearman correlation against "catB" (Known Answer #6)
    res = spearman_filter_tool.invoke({"target_class": "catB"})
    full_report = session.results_store[res["result_id"]]["full_report"]
    correlations = dict(zip(full_report["term"], full_report["spearman_r"]))

    # Assertions against Known Answer #6 Spearman r values
    assert correlations["alpha"] == pytest.approx(-0.7500, abs=1e-3)
    assert correlations["always"] == pytest.approx(0.3780, abs=1e-3)
    assert correlations["beta"] == pytest.approx(-0.7746, abs=1e-3)
    assert correlations["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert correlations["gamma"] == pytest.approx(0.9363, abs=1e-3)

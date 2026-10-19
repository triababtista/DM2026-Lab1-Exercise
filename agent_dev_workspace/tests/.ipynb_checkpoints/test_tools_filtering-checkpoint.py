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

    assert "always" in res["removed_terms"]
    assert sorted(res["kept_terms"]) == sorted(["alpha", "beta", "delta", "gamma"])
    assert res["n_kept_terms"] == 4
    assert res["n_removed_terms"] == 1

    # Check full report variances against known answer #5
    full_report = session.results_store[res["result_id"]]["full_report"]
    variance_dict = dict(zip(full_report["term"], full_report["variance"]))

    assert variance_dict["alpha"] == pytest.approx(1.1875, rel=1e-3)
    assert variance_dict["always"] == pytest.approx(0.109375, rel=1e-3)
    assert variance_dict["beta"] == pytest.approx(0.234375, rel=1e-3)
    assert variance_dict["delta"] == pytest.approx(0.25, rel=1e-3)
    assert variance_dict["gamma"] == pytest.approx(1.109375, rel=1e-3)

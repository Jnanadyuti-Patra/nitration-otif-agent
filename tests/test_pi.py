import json

from otifagent import pi, simulate
from otifagent.agent import search_sop


def test_every_tool_returns_json():
    for f in pi.TOOLS:
        json.dumps(f(), default=str)


def test_heat_balance_recovers_design_U_on_clean_reactor():
    u = pi.jacket_heat_transfer()["median_U_W_per_m2K"]["2026-01"]
    assert abs(u["R-101"] - simulate.U_CLEAN) < 15


def test_fouled_reactor_is_visible_in_back_calculated_U():
    u = pi.jacket_heat_transfer()["median_U_W_per_m2K"]["2026-06"]
    assert u["R-103"] < 0.7 * u["R-101"]


def test_sop_retrieval_finds_fouling_procedure():
    assert "SOP-M07" in search_sop("jacket fouling cleaning")["results"][0]

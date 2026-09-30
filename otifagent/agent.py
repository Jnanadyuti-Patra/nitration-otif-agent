"""LangGraph workflow: planner -> tools -> analyst -> SOP retrieval -> verifier.

A 3B local model will not reliably run a long ReAct loop, so the graph gives each
LLM call one narrow job and checks its work in code:
  planner   LLM chooses which process-intelligence tools to call (tool calling)
  tools     runs a fixed diagnostic panel plus the drill-downs the planner chose
  analyst   LLM writes a ranked diagnosis as a pydantic-validated JSON object
  sop       TF-IDF retrieval attaches the matching plant SOP to each root cause (RAG)
  verifier  code rejects any number the analyst quotes that no tool returned,
            and sends the draft back once with the list of unsupported numbers

Run:  python -m otifagent.agent "Why did our export OTIF drop from April?"
Model: local Ollama, default qwen2.5:3b (set OTIF_MODEL to change).
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import sys
import time
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import pi

KB = pathlib.Path(__file__).resolve().parents[1] / "knowledge" / "plant_sops.md"
_SECTIONS = ["## " + s for s in KB.read_text(encoding="utf-8").split("## ")[1:]]
_VEC = TfidfVectorizer(stop_words="english").fit(_SECTIONS)
_MAT = _VEC.transform(_SECTIONS)


def search_sop(query: str) -> dict:
    """Retrieve the 2 most relevant plant SOP sections (safety limits, fixes, specs) for a query."""
    sims = cosine_similarity(_VEC.transform([query]), _MAT)[0]
    return {"results": [_SECTIONS[i].strip() for i in sims.argsort()[::-1][:2]]}


TOOLS = {f.__name__: f for f in pi.TOOLS}
# standard diagnostic panel (like a process app's fixed KPIs); the planner LLM adds drill-downs
MIN_EVIDENCE = ["otif_by", "order_lead_time_breakdown", "jacket_heat_transfer", "rework_rate_by",
                "qc_wait_by_shift", "value_at_stake"]


class RootCause(BaseModel):
    rank: int
    driver: str = Field(description="plain-English cause naming the equipment, supplier, shift or season involved, e.g. 'Weekend packaging line shortage'")
    evidence: str = Field(description="the tool numbers that prove it")
    impact_lakh_inr: float | None = Field(None, description="rupee impact from value_at_stake, in lakh")


class Diagnosis(BaseModel):
    summary: str
    root_causes: list[RootCause]


class State(TypedDict, total=False):
    question: str
    planned: list[str]
    evidence: dict
    diagnosis: dict
    unsupported: list[str]
    rounds: int
    sops: dict


PLANNER = """You are the planner for a process-intelligence analyst at a batch nitration plant \
(Dahej, Gujarat; 3 reactors R-101..R-103; export orders shipped via Mundra). \
Call every tool needed to explain the manager's question. Candidate drivers to test: reactor \
capacity and queueing, jacket fouling versus warm cooling water (jacket_heat_transfer), raw-material \
supplier quality (rework_rate_by supplier), QC laboratory delays (qc_wait_by_shift), and the rupee \
impact (value_at_stake). Call several tools at once."""

ANALYST = """You are a process-intelligence analyst. Using ONLY the JSON evidence below, explain the \
manager's question. Rank the root causes by rupee impact. Every number you write must appear in the \
evidence. Compare months and reactors explicitly (for example Jan versus May, R-103 versus R-101). \
Test every driver in the evidence: cooling-water temperature by month, jacket U by reactor versus \
the design value, rework rate by supplier, QC turnaround by shift. Report each driver that the \
evidence supports, with its rupee impact where value_at_stake gives one."""


def _llm(model: str | None):
    from langchain_ollama import ChatOllama
    return ChatOllama(model=model or os.getenv("OTIF_MODEL", "qwen2.5:3b"), temperature=0, num_ctx=12288)


def _numbers(text: str) -> set[float]:
    return {float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*\.?\d*", text)}


def build(model: str | None = None):
    llm = _llm(model)
    tools = [StructuredTool.from_function(f) for f in pi.TOOLS]

    def planner(s: State) -> State:
        msg = llm.bind_tools(tools).invoke([SystemMessage(PLANNER), HumanMessage(s["question"])])
        return {"planned": [dict(name=c["name"], args=c["args"]) for c in msg.tool_calls]}

    def run_tools(s: State) -> State:
        calls = [c for c in s["planned"] if c["name"] in TOOLS]
        names = {c["name"] for c in calls}
        calls += [dict(name=n, args={}, guardrail=True) for n in MIN_EVIDENCE if n not in names]
        ev = {}
        for c in calls:
            key = c["name"] + (json.dumps(c["args"], sort_keys=True) if c["args"] else "")
            try:
                ev[key] = TOOLS[c["name"]](**c["args"])
            except Exception:  # invalid arguments from the LLM fall back to the tool defaults
                key = c["name"]
                ev[key] = TOOLS[c["name"]]()
        return {"evidence": ev, "rounds": 0}

    def analyst(s: State) -> State:
        prompt = ANALYST + "\n\nEVIDENCE:\n" + json.dumps(s["evidence"], default=str)
        if s.get("unsupported"):
            prompt += ("\n\nYour previous draft quoted numbers that are not in the evidence: "
                       + ", ".join(s["unsupported"]) + ". Use only evidence numbers.")
        d = llm.with_structured_output(Diagnosis).invoke(
            [SystemMessage(prompt), HumanMessage(s["question"])])
        return {"diagnosis": d.model_dump(), "rounds": s.get("rounds", 0) + 1}

    def verifier(s: State) -> State:
        allowed = _numbers(json.dumps(s["evidence"], default=str))
        text = json.dumps(s["diagnosis"])
        bad = []
        for x in _numbers(text):
            if x in (0, 1, 2, 3, 4, 5) or x in allowed:  # ranks, small counts
                continue
            if not any(abs(x - a) <= max(0.051, 0.005 * abs(a)) for a in allowed):
                bad.append(f"{x:g}")
        return {"unsupported": sorted(set(bad))}

    def attach_sops(s: State) -> State:
        return {"sops": {rc["driver"]: search_sop(rc["driver"] + " " + rc["evidence"])["results"][0].split("\n")[0]
                         for rc in s["diagnosis"]["root_causes"]}}

    g = StateGraph(State)
    for name, fn in [("planner", planner), ("tools", run_tools), ("analyst", analyst),
                     ("verifier", verifier), ("sop", attach_sops)]:
        g.add_node(name, fn)
    g.set_entry_point("planner")
    g.add_edge("planner", "tools")
    g.add_edge("tools", "analyst")
    g.add_edge("analyst", "verifier")
    g.add_conditional_edges("verifier", lambda s: "analyst" if s["unsupported"] and s["rounds"] < 2 else "sop")
    g.add_edge("sop", END)
    return g.compile()


def ask(question: str, model: str | None = None) -> dict:
    t0 = time.time()
    s = build(model).invoke({"question": question})
    return {"question": question,
            "tools_chosen_by_llm": [c["name"] for c in s["planned"]],
            "tools_run": list(s["evidence"]),
            "analyst_rounds": s["rounds"],
            "unsupported_numbers_left": s["unsupported"],
            "diagnosis": s["diagnosis"], "sops": s["sops"],
            "seconds": round(time.time() - t0, 1)}


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Why did our export OTIF drop from April, and what is it costing us?"
    print(json.dumps(ask(q), indent=2, ensure_ascii=False))


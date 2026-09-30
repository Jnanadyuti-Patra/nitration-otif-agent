"""Process-intelligence tools over the object-centric event log.

Every function returns a small, JSON-friendly dict so an LLM agent can call it
as a tool and quote the numbers back. Nothing here knows the hidden root causes.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from . import simulate as sim

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"

# Commercial assumptions (stated, not sourced from any company)
ASSUMPTIONS = {
    "ncb_price_inr_per_kg": 120.0,
    "contribution_margin_inr_per_kg": 45.0,
    "rework_wash_cost_inr": 25_000.0,
    "late_delivery_penalty_pct_of_order_value": 2.0,
}


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    ev = pd.read_csv(DATA / "events.csv", parse_dates=["timestamp"])
    od = pd.read_csv(DATA / "orders.csv", parse_dates=["created", "due", "dispatched"])
    return ev, od


EV, OD = load() if (DATA / "events.csv").exists() else (None, None)


def _batches() -> pd.DataFrame:
    """One row per batch with step durations (hours) and attributes."""
    ev = EV[EV.batch_id.notna()]
    first = ev.pivot_table(index="batch_id", columns="activity", values="timestamp", aggfunc="first")
    last = ev.pivot_table(index="batch_id", columns="activity", values="timestamp", aggfunc="last")
    attrs = ev.groupby("batch_id").agg(order_id=("order_id", "first"), reactor=("reactor", "first"),
                                       supplier=("supplier", "first"), lot_id=("lot_id", "first"))
    b = attrs.copy()
    h = lambda a, c: (c - a).dt.total_seconds() / 3600
    b["dosing_h"] = h(first["Start Acid Dosing"], first["End Acid Dosing"])
    b["charge_to_transfer_h"] = h(first["Charge Chlorobenzene"], first["Transfer to Hold Tank"])
    b["transfer_to_release_h"] = h(first["Transfer to Hold Tank"], last["Release Batch"])
    b["qc_fails"] = ev[ev.activity == "QC Result: Fail"].groupby("batch_id").size()
    b["qc_fails"] = b.qc_fails.fillna(0).astype(int)
    b["start"] = first["Charge Chlorobenzene"]
    b["month"] = b.start.dt.strftime("%Y-%m")
    return b.dropna(subset=["dosing_h", "transfer_to_release_h"])


def otif_by(dimension: str = "month") -> dict:
    """On-time-in-full rate of export orders by 'month' or 'customer'."""
    od = OD.copy()
    od["month"] = od.created.dt.strftime("%Y-%m")
    g = od.groupby(dimension).agg(orders=("order_id", "count"), otif_pct=("on_time", "mean"))
    g["otif_pct"] = (g.otif_pct * 100).round(1)
    return {"overall_otif_pct": round(od.on_time.mean() * 100, 1), "by_" + dimension: g.to_dict("index")}


def order_lead_time_breakdown() -> dict:
    """Where an order's time goes: waiting for a reactor, in the reactor, QC+rework, dispatch (median hours, by month)."""
    b = _batches()
    ev = EV
    created = OD.set_index("order_id").created
    b["wait_for_reactor_h"] = (b.start - b.order_id.map(created)).dt.total_seconds() / 3600
    out = b.groupby("month")[["wait_for_reactor_h", "charge_to_transfer_h", "transfer_to_release_h"]].median()
    return {"median_hours_by_month": out.round(1).to_dict("index"),
            "note": "wait_for_reactor = order created to batch charged; transfer_to_release = QC + rework"}


def step_duration_by(step: str = "dosing_h", dimension: str = "reactor") -> dict:
    """Median duration (h) of a step, split by a dimension.
    step: dosing_h | charge_to_transfer_h | transfer_to_release_h
    dimension: reactor | month | supplier | reactor_month"""
    b = _batches()
    if dimension == "reactor_month":
        t = b.pivot_table(index="month", columns="reactor", values=step, aggfunc="median").round(2)
        return {"step": step, "median_hours": t.to_dict("index")}
    return {"step": step, "median_hours": b.groupby(dimension)[step].median().round(2).to_dict(),
            "batches": b.groupby(dimension).size().to_dict()}


def rework_rate_by(dimension: str = "supplier") -> dict:
    """Share of batches that failed QC at least once (and needed a rework wash), by supplier | reactor | month."""
    b = _batches()
    g = b.groupby(dimension).agg(batches=("qc_fails", "size"), rework_pct=("qc_fails", lambda s: (s > 0).mean() * 100))
    return {"rework_pct": g.round(1).to_dict("index")}


def qc_wait_by_shift() -> dict:
    """Median hours from 'Take QC Sample' to its QC result, by the shift the sample was taken in."""
    ev = EV[EV.activity.isin(["Take QC Sample", "QC Result: Pass", "QC Result: Fail"])].sort_values(["timestamp", "event_id"])
    rows = []
    for _, g in ev.groupby("batch_id"):
        pending = None
        for r in g.itertuples():
            if r.activity == "Take QC Sample":
                pending = r
            elif pending is not None:
                rows.append((pending.shift, (r.timestamp - pending.timestamp).total_seconds() / 3600))
                pending = None
    d = pd.DataFrame(rows, columns=["shift", "h"])
    return {"median_qc_turnaround_h": d.groupby("shift").h.median().round(2).to_dict(),
            "samples": d.groupby("shift").size().to_dict()}


def jacket_heat_transfer(reactor: str = "all") -> dict:
    """Back-calculate each reactor's jacket U (W/m2K) from dosing times in the event log.
    Dosing is heat-removal limited: U = n_CB*|dH|/(t_dose*A*dT). A falling U means fouling;
    a longer dosing time with steady U means warmer cooling water (smaller dT)."""
    b = _batches()
    ev = EV[EV.activity == "Start Acid Dosing"].set_index("batch_id")
    b["cw_supply_c"] = ev.cw_supply_c
    n = sim.BATCH_CB_KG * 1000 / sim.MW_CB
    dT = sim.T_REACTOR - (b.cw_supply_c + sim.JACKET_RISE / 2)
    b["U"] = n * sim.DH_R / (b.dosing_h * 3600 * sim.AREA * dT)
    if reactor != "all":
        b = b[b.reactor == reactor]
    u = b.pivot_table(index="month", columns="reactor", values="U", aggfunc="median").round(0)
    cw = b.groupby("month").cw_supply_c.first()
    return {"median_U_W_per_m2K": u.to_dict("index"), "cw_supply_c_by_month": cw.to_dict(),
            "design_U_clean": sim.U_CLEAN}


def process_variants(top: int = 5) -> dict:
    """Most common activity sequences per batch (process variants) with counts."""
    ev = EV[EV.batch_id.notna()].sort_values(["timestamp", "event_id"])
    seq = ev.groupby("batch_id").activity.agg(lambda s: " > ".join(s))
    vc = seq.value_counts()
    return {"total_batches": int(vc.sum()),
            "variants": [{"count": int(c), "pct": round(c / vc.sum() * 100, 1), "path": p} for p, c in vc.head(top).items()]}


def value_at_stake() -> dict:
    """Rupee impact of each operational driver over the log period, using ASSUMPTIONS."""
    a = ASSUMPTIONS
    b = _batches()
    kg_ncb = sim.BATCH_CB_KG / sim.MW_CB * sim.MW_NCB * sim.YIELD
    margin_batch = kg_ncb * a["contribution_margin_inr_per_kg"]
    clean_cycle_h = b[(b.reactor != "R-103") & (b.month == "2026-01")].charge_to_transfer_h.median() + 1.0
    per_reactor_h = margin_batch / clean_cycle_h
    # expected dosing time if every reactor had a clean jacket in the same month
    ev = EV[EV.activity == "Start Acid Dosing"].set_index("batch_id")
    cw = ev.cw_supply_c.reindex(b.index)
    n = sim.BATCH_CB_KG * 1000 / sim.MW_CB
    t_clean = n * sim.DH_R / (sim.U_CLEAN * sim.AREA * (sim.T_REACTOR - (cw + sim.JACKET_RISE / 2))) / 3600
    t_jan = n * sim.DH_R / (sim.U_CLEAN * sim.AREA * (sim.T_REACTOR - (sim.CW_SUPPLY[1] + sim.JACKET_RISE / 2))) / 3600
    fouling_h = (b.dosing_h - t_clean).clip(lower=0)[b.reactor == "R-103"].sum()
    summer_h = (t_clean - t_jan).clip(lower=0).sum()
    rework_n = int((b.qc_fails > 0).sum())
    late = OD[~OD.on_time]
    late_val = (late.qty_kg * a["ncb_price_inr_per_kg"]).sum() * a["late_delivery_penalty_pct_of_order_value"] / 100
    lakh = lambda x: round(float(x) / 1e5, 1)
    return {
        "assumptions": a,
        "margin_per_reactor_hour_inr": round(per_reactor_h),
        "R-103_excess_dosing_hours": round(float(fouling_h), 1),
        "R-103_fouling_lost_margin_lakh_inr": lakh(fouling_h * per_reactor_h),
        "summer_cooling_water_excess_hours_all_reactors": round(float(summer_h), 1),
        "summer_cw_lost_margin_lakh_inr": lakh(summer_h * per_reactor_h),
        "rework_batches": rework_n,
        "rework_cost_lakh_inr": lakh(rework_n * a["rework_wash_cost_inr"]),
        "late_orders": int(len(late)),
        "late_delivery_penalties_lakh_inr": lakh(late_val),
    }


TOOLS = [otif_by, order_lead_time_breakdown, step_duration_by, rework_rate_by, qc_wait_by_shift,
         jacket_heat_transfer, process_variants, value_at_stake]


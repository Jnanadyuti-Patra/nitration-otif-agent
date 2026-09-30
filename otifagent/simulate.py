"""Object-centric event log for a batch nitration plant at Dahej, Gujarat.

Three jacketed reactors nitrate chlorobenzene (CB) with mixed acid to make
nitrochlorobenzene (NCB) for export orders shipped through Mundra.

The chemistry sets the clock: dosing is heat-removal limited, so every batch's
dosing time is t = n_CB * |dH_r| / (U * A * dT_lm). That is how jacket fouling
and the Gujarat summer cooling-water temperature end up in the event log.

Hidden root causes (the agent is never told these; eval.py scores against them):
  1. R-103 jacket fouling from 1 Feb: U decays 450 -> ~250 W/m2K by June.
  2. Summer cooling water: CW supply rises 24 C (Jan) -> 32 C (May/Jun).
  3. Supplier B chlorobenzene lots carry excess moisture: QC fails 20% vs 4%.
  4. Night-shift QC: one analyst covers two labs, so samples wait longer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import simpy

START = pd.Timestamp("2026-01-01")
DAYS = 181  # Jan to Jun 2026

# --- physical basis -------------------------------------------------------
BATCH_CB_KG = 2500.0
MW_CB, MW_NCB = 112.56, 157.55
DH_R = 117e3            # J/mol, benzene mononitration used as proxy (exothermic)
T_REACTOR = 50.0        # C, dosing temperature
AREA = 12.0             # m2 jacket area
U_CLEAN = 450.0         # W/m2K
JACKET_RISE = 5.0       # K cooling-water rise across the jacket
YIELD = 0.97
# CW supply = wet-bulb + 4 K approach (assumed monthly Dahej wet-bulb values)
CW_SUPPLY = {1: 24.0, 2: 25.0, 3: 27.0, 4: 30.0, 5: 32.0, 6: 32.0}

ORDER_RATE = 2.0  # orders/day
REACTORS = ["R-101", "R-102", "R-103"]
FOUL_START = pd.Timestamp("2026-02-01")


def u_value(reactor: str, t: pd.Timestamp) -> float:
    if reactor != "R-103" or t < FOUL_START:
        return U_CLEAN
    days = (t - FOUL_START).days
    return 250.0 + 200.0 * np.exp(-days / 60.0)


def dosing_hours(reactor: str, t: pd.Timestamp, rng) -> float:
    n = BATCH_CB_KG * 1000 / MW_CB                    # mol CB
    t_cw = CW_SUPPLY[t.month] + JACKET_RISE / 2        # mean jacket temperature
    duty = u_value(reactor, t) * AREA * (T_REACTOR - t_cw)   # W
    return n * DH_R / duty / 3600 * rng.lognormal(0, 0.05)


def shift_of(t: pd.Timestamp) -> str:
    return "Night" if (t.hour >= 22 or t.hour < 6) else ("Day" if t.hour < 14 else "Evening")


def simulate(seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (events, orders). Each event row is linked to its objects."""
    rng = np.random.default_rng(seed)
    env = simpy.Environment()
    ts = lambda: START + pd.Timedelta(hours=env.now)
    events, orders = [], []
    queue = simpy.Store(env)
    qc_lab = simpy.Resource(env, capacity=1)
    order_done = {}
    customers = ["BASF-SG", "UPL-Export", "Sumitomo-JP", "Bayer-DE", "Syngenta-CH"]

    def log(activity, batch=None, order=None, reactor=None, lot=None, supplier=None, **kw):
        t = ts()
        events.append(dict(timestamp=t, activity=activity, batch_id=batch, order_id=order,
                           reactor=reactor, lot_id=lot, supplier=supplier, shift=shift_of(t), **kw))

    def order_source():
        i = 0
        while env.now < DAYS * 24 - 240:
            yield env.timeout(rng.exponential(24 / ORDER_RATE))
            i += 1
            oid = f"SO-{i:04d}"
            nb = int(rng.integers(1, 5))
            due = ts() + pd.Timedelta(days=int(rng.choice([5, 6, 7, 9])))  # vessel cut-off at Mundra
            orders.append(dict(order_id=oid, customer=str(rng.choice(customers)),
                               batches=nb, qty_kg=round(nb * BATCH_CB_KG / MW_CB * MW_NCB * YIELD),
                               created=ts(), due=due))
            order_done[oid] = {"left": nb}
            log("Create Sales Order", order=oid)
            for b in range(nb):
                yield queue.put((oid, f"{oid}-B{b + 1}"))

    def qc_and_release(oid, bid, reactor, lot, supplier):
        attempt = 0
        while True:
            attempt += 1
            log("Take QC Sample", bid, oid, reactor, lot, supplier)
            night = shift_of(ts()) == "Night"
            with qc_lab.request() as req:
                yield req
                if night:
                    yield env.timeout(rng.gamma(2.0, 2.0))    # analyst on the other lab
                yield env.timeout(rng.normal(1.5, 0.2))
            p_fail = 0.20 if supplier == "Supplier B" else 0.04
            if rng.random() < p_fail:
                log("QC Result: Fail", bid, oid, reactor, lot, supplier, attempt=attempt)
                log("Start Rework Wash", bid, oid, reactor, lot, supplier)
                yield env.timeout(rng.normal(8.0, 1.0))
                log("End Rework Wash", bid, oid, reactor, lot, supplier)
                continue
            log("QC Result: Pass", bid, oid, reactor, lot, supplier, attempt=attempt)
            break
        log("Release Batch", bid, oid, reactor, lot, supplier)
        order_done[oid]["left"] -= 1
        if order_done[oid]["left"] == 0:
            yield env.timeout(rng.normal(20, 3))                  # pack + truck to Mundra
            log("Dispatch Order", order=oid)

    def reactor_proc(name):
        lot_no = 0
        while True:
            oid, bid = yield queue.get()
            lot_no += 1
            supplier = "Supplier B" if rng.random() < 0.4 else "Supplier A"
            lot = f"CB-{name[-3:]}-{lot_no:03d}"
            log("Charge Chlorobenzene", bid, oid, name, lot, supplier)
            yield env.timeout(rng.normal(1.5, 0.15))
            t0 = ts()
            log("Start Acid Dosing", bid, oid, name, lot, supplier,
                cw_supply_c=CW_SUPPLY[t0.month])
            yield env.timeout(dosing_hours(name, t0, rng))
            log("End Acid Dosing", bid, oid, name, lot, supplier)
            yield env.timeout(rng.normal(1.0, 0.1))
            log("Quench and Separate", bid, oid, name, lot, supplier)
            yield env.timeout(rng.normal(2.0, 0.2))
            log("Transfer to Hold Tank", bid, oid, name, lot, supplier)
            env.process(qc_and_release(oid, bid, name, lot, supplier))
            yield env.timeout(rng.normal(1.0, 0.1))              # rinse before next charge

    env.process(order_source())
    for r in REACTORS:
        env.process(reactor_proc(r))
    env.run(until=DAYS * 24)

    ev = pd.DataFrame(events).sort_values("timestamp", kind="stable").reset_index(drop=True)
    ev.insert(0, "event_id", [f"E{i:06d}" for i in range(len(ev))])
    od = pd.DataFrame(orders)
    disp = ev[ev.activity == "Dispatch Order"].set_index("order_id").timestamp
    od["dispatched"] = od.order_id.map(disp)
    od["on_time"] = od.dispatched.notna() & (od.dispatched <= od.due)
    return ev, od


if __name__ == "__main__":
    import pathlib
    out = pathlib.Path(__file__).resolve().parents[1] / "data"
    out.mkdir(exist_ok=True)
    ev, od = simulate()
    ev.to_csv(out / "events.csv", index=False)
    od.to_csv(out / "orders.csv", index=False)
    print(len(ev), "events,", len(od), "orders, OTIF", round(od.on_time.mean() * 100, 1), "%")
    print(od.assign(m=od.created.dt.month).groupby("m").on_time.mean().round(3).to_dict())





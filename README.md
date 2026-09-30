# Nitration OTIF Agent

An AI agent that explains why a Gujarat specialty-chemical plant is missing export
deliveries, using process intelligence (an object-centric event log) as its context
instead of guesswork.

## Problem statement

A batch nitration plant at Dahej, Gujarat makes nitrochlorobenzene (NCB), a dye and
agrochemical intermediate, for export customers shipped through Mundra port. Its
on-time-in-full (OTIF) rate was about 100% in Q1 2026 and fell to 76% in April and
26% in May. The plant managers get the late orders, not the reasons. Gujarat summers
make cooling water warmer, reactors foul, raw-material lots vary by supplier, and the QC
lab is thin on nights, but nobody knows which of these is driving the miss or what each
one costs.

Nitration is exothermic and its dosing is heat-removal limited. Operators cannot "just
dose faster" without risking a runaway (SOP-N01), so a slow jacket is a capacity
problem, not a setpoint problem.

## What is in the repo

| Part | File | What it does |
|---|---|---|
| Plant simulator | `otifagent/simulate.py` | SimPy model of 3 jacketed reactors, a QC lab, rework and dispatch. It writes an object-centric event log in which each event is linked to its order, batch, reactor and raw-material lot. Each dosing time comes from a jacket heat balance, `t = n_CB * dH / (U * A * dT)`. |
| Process-intelligence tools | `otifagent/pi.py` | 8 functions the agent can call: OTIF by month or customer, lead-time breakdown, step durations, rework by supplier, QC turnaround by shift, jacket U back-calculated from event timestamps, process variants, and rupee value at stake. |
| Knowledge base | `knowledge/plant_sops.md` | Plant SOPs (illustrative), retrieved with TF-IDF (RAG). |
| Agent | `otifagent/agent.py` | A LangGraph workflow: planner LLM (tool calling) -> tools with a minimum-evidence guardrail -> analyst LLM (pydantic-validated diagnosis) -> numeric verifier -> SOP retrieval. |
| Evaluation | `otifagent/eval.py` | Scores the diagnosis against the 4 root causes hidden in the simulator. |

The model runs locally through Ollama (default `qwen2.5:3b`, no API key). A 3B model
will not reliably run a long ReAct loop (the first version stopped after 2 tool calls),
so the graph gives each LLM call one narrow job and the verifier rejects any number that
no tool returned.

## Hidden root causes (ground truth)

1. R-103 jacket fouling from February: back-calculated U falls from 451 to 273 W/m2K while R-101 and R-102 stay near 450.
2. Summer cooling water: supply rises from 24 C in January to 32 C in May and June, cutting the jacket driving force on every reactor.
3. Supplier B chlorobenzene: 16.5% of its batches need a rework wash, versus 4.3% for Supplier A.
4. Night-shift QC: median sample turnaround is 8.1 h at night versus 1.9 to 2.6 h on other shifts.

Value at stake over Jan to Jun 2026, under the stated assumptions in `pi.ASSUMPTIONS`
(margin INR 45/kg, rework INR 25,000 per wash, 2% late-delivery penalty):
summer cooling water about INR 162 lakh of lost reactor margin, R-103 fouling about
INR 85 lakh, rework INR 18.5 lakh, late-delivery penalties INR 16.4 lakh.

## Results

Evaluation on `qwen2.5:3b`, CPU only, 3 phrasings of the manager's question (`results/eval.json`):

| Question | Tools the planner LLM chose | Hidden root causes found | Numbers grounded |
|---|---|---|---|
| Why did our export OTIF drop from April, and what is it costing us? | 5 | 3 of 4 | yes |
| Customers say shipments are late since summer; what should we fix first? | 4 | 2 of 4 | no, verifier flagged 6 |
| Where is lead time being lost, and how much money is at stake? | 1 | 0 of 4 | no, verifier flagged 2 |

Overall root-cause recall is 5 of 12 (0.42), and each run takes about 3 minutes on a laptop
CPU. The tools themselves isolate all four causes (see above); the weak link is the 3B
analyst turning the evidence into a ranked story, and the verifier is what exposes it. The
first ReAct-only version stopped after 2 tool calls, which is why the workflow is a graph.
Next steps: a larger model (`OTIF_MODEL=qwen2.5:7b`), and feeding the verifier's rejections
back as a checklist rather than a list of numbers.

## Run it

```bash
pip install -r requirements.txt
ollama pull qwen2.5:3b
python -m otifagent.simulate      # writes data/events.csv and data/orders.csv
python -m otifagent.agent "Why did our export OTIF drop from April?"
python -m otifagent.eval          # writes results/eval.json
pytest -q
```

## Limits

The plant, its data and its SOPs are simulated. No company's data is used. The nitration
enthalpy uses the benzene value (-117 kJ/mol) as a proxy, cooling-water temperatures are
assumed from typical Dahej wet-bulb values plus a 4 K tower approach, and the commercial
figures are assumptions. The point is the method: turn process data into an event log,
give an agent tools over it, and make it prove every number.


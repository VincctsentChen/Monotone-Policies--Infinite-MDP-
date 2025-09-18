# Infinite-Horizon Markov Decision Process for Disease Progression

This repository implements an **infinite-horizon Markov Decision Process (MDP)** to model patients’ disease progression under different treatments. The repo builds the transition probabilities between different disease states, reward, return (total reward), and possible treatment options (drug types). The return of the MDP is the Quality-adjusted life years (QALYs) for each patient.

The model produces a **transition probability tensor** of shape **(10, 10, 6)**:
- **10** current states (rows)
- **10** next states (columns)
- **6** actions (treatments)

This framework enables simulation of long-term treatment policies for chronic disease management.

---

## Treatments

The six treatments are defined in `test_infinite_horizon_with_data.py`:

- **T0: `NT`** — No Treatment  
- **T1: `ACE`** — ACE Inhibitors  
- **T2: `ARB`** — Angiotensin Receptor Blockers  
- **T3: `BB`** — Beta Blockers  
- **T4: `CCB`** — Calcium Channel Blockers  
- **T5: `TH`** — Thiazide Diuretics  

> Source snippet (context only):
> ```python
> alldrugs = ["ACE", "ARB", "BB", "CCB", "TH"]  # base list
> drugs.insert(0, "NT")                         # add no-treatment option
> alldrugs = drugs                              # simplified: single-drug actions
> ```

---

## Getting Started

The main entry point is **`test_infinite_horizon_with_data.py`**.

1) **Clone** the repo and enter the project directory:
```bash
git clone <YOUR_REPO_URL>.git
cd <YOUR_REPO_NAME>

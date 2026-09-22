# Contributing to agent-ab-tester

## Setup
```bash
git clone https://github.com/daniellopez882/agent-ab-tester.git
cd agent-ab-tester
pip install -e ".[dev,all]"
pytest tests/ -v
```

## High-Impact Contributions
- **Bayesian analysis** — posterior probability that treatment > control
- **Multi-armed bandit** — sequential allocation to minimize regret
- **HTML report** — shareable experiment report with charts
- **Experiment registry** — track all experiments in a local DB
- **Stratified analysis** — break down results by task category
- **Auto sample-size** — stop early when significance is reached

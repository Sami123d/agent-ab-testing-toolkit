<div align="center">

# agent-ab-tester

**A/B test your AI agents with statistical rigor.**

Run two agent versions on the same tasks. Get a statistically significant answer:
*"Did the new prompt actually improve quality, or was it just noise?"*

[![PyPI](https://img.shields.io/badge/pypi-v0.1.0-blue?style=flat-square)](https://pypi.org/project/agent-ab-tester/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/downloads/)

[Quick Start](#-quick-start) · [How It Works](#-how-it-works) · [Statistics](#-statistical-methods) · [Dashboard](#-dashboard) · [CLI](#-cli)

</div>

---

## 🔥 The Problem

You changed your agent's system prompt. Output *looks* better. But is it **actually** better — or did you just happen to test on easy queries?

> *"32% of organizations cite quality as the top barrier to agent deployment."*
> — LangChain State of Agent Engineering, 2026

Agent teams make deployment decisions on vibes — running 5 examples, eyeballing the output, shipping. Every real A/B testing platform costs $10K+/year (Maxim, Statsig, Optimizely) or tests web pages, not agent pipelines.

**agent-ab-tester brings statistical rigor to agent development.** Point it at two agent versions, give it tasks, get a p-value.

---

## ⚡ Quick Start

```bash
pip install agent-ab-tester
```

```python
# my_agents.py
from agent_ab_tester import AgentVariant

class BaselineAgent(AgentVariant):
    async def invoke(self, task: str) -> str:
        return await my_current_agent(task)

class ChallengerAgent(AgentVariant):
    async def invoke(self, task: str) -> str:
        return await my_new_agent(task)
```

```bash
agent-ab run \
  --control my_agents:BaselineAgent \
  --treatment my_agents:ChallengerAgent \
  --tasks tasks.jsonl \
  --num-tasks 100

# ┌──────────────────────────────────────────────────────────┐
# │  Metric          │ Control │ Treatment │  Δ     │ p-value │
# │  Quality (judge) │ 7.2/10  │ 8.1/10    │ +12.5% │ 0.003** │
# │  Cost per run    │ $0.042  │ $0.038    │  -9.5% │ 0.041*  │
# │  Latency         │  4.2s   │  4.8s     │ +14.3% │ 0.012*  │
# ├──────────────────────────────────────────────────────────┤
# │  Verdict: ✅ Treatment significantly better on quality   │
# │           ⚠️ But 14.3% slower                           │
# └──────────────────────────────────────────────────────────┘
```

---

## 🧬 How It Works

```
    Task Pool (100 tasks)
          │
    ┌─────┴─────┐
    ▼           ▼
 CONTROL     TREATMENT
 (Agent A)   (Agent B)
    │           │
    ▼           ▼
  LLM Judge (blind scoring — doesn't know which is which)
    │
    ▼
  Statistical Tests
  • Bayesian posterior sampling (P(B > A))
  • Sequential Probability Ratio Test (SPRT)
  • Welch's t-test (paired)
  • Effect size (Cohen's d)
  • 95% Confidence intervals
  • Bonferroni correction
    │
    ▼
  High-Fidelity Dashboard & Report
```

**Key design decisions:**
- **Blind judging** — LLM judge doesn't know which output is A vs B (eliminates position bias)
- **Paired design** — Both agents answer the *same* tasks, controlling for task difficulty
- **Multiple metrics** — Quality, cost, latency tracked independently with tradeoff analysis
- **Conservative defaults** — 95% confidence, exact p-values, no "vibes"

---

## 📊 Metrics

| Metric | How Measured | Direction |
|--------|-------------|-----------|
| **Quality** | LLM judge scores 0-10 (blind) | Higher = better |
| **Cost** | Tokens × model pricing | Lower = better |
| **Latency** | Wall-clock time | Lower = better |
| **Tokens** | Input + output count | Lower = better |
| **Error rate** | Failed invocations / total | Lower = better |

### Custom Metrics

```python
experiment = Experiment(
    control=V1(), treatment=V2(),
    custom_metrics=[
        {"name": "citations", "fn": count_citations, "direction": "higher"},
        {"name": "word_count", "fn": lambda o: len(o.split()), "direction": "higher"},
    ],
)
```

---

## 📐 Statistical Methods

### Welch's t-test (default)

Paired differences per task, test whether mean difference ≠ 0.

```
H₀: μ_treatment - μ_control = 0
H₁: μ_treatment - μ_control ≠ 0
```

Reports: p-value, 95% CI, Cohen's d effect size, Bonferroni correction.

### Bootstrap (for small samples)

```bash
agent-ab run --method bootstrap --bootstrap-samples 10000
```

### Power Analysis

```bash
agent-ab power --effect-size 0.5 --alpha 0.05 --power 0.80
# → Minimum tasks: 64 per variant
```

---

## 🖥️ CLI

```bash
# Basic experiment
agent-ab run --control agents:V1 --treatment agents:V2 --tasks tasks.jsonl

# With options
agent-ab run --control agents:V1 --treatment agents:V2 \
  --tasks tasks.jsonl --num-tasks 100 \
  --confidence 0.99 --judge-model gpt-4o \
  --format json -o results.json

# CI/CD: fail if treatment isn't significantly better
agent-ab run --control agents:V1 --treatment agents:V2 \
  --tasks tasks.jsonl --require-significant quality --exit-code

# Power analysis
agent-ab power --effect-size 0.5

# List available tasks
agent-ab tasks list
```

---

## 🆚 How This Compares

| Tool | Open Source | Agent-Specific | Stats Rigor | CLI-First | Cost |
|------|:-:|:-:|:-:|:-:|------|
| **agent-ab-tester** | ✅ | ✅ | ✅ | ✅ | Free |
| Maxim AI | ❌ | ✅ | ✅ | ❌ | $$$$ |
| Statsig | ❌ | ❌ | ✅ | ❌ | $$$$ |
| GrowthBook | ✅ | ❌ | ✅ | ❌ | Free |
| LangSmith | ❌ | Partial | Partial | ❌ | $$$ |
| Promptfoo | ✅ | ❌ | ❌ | ✅ | Free |

**The key gap:** GrowthBook and Statsig A/B test web pages. Promptfoo tests prompts but without statistical tests. LangSmith compares experiments side-by-side but without p-values. **Nobody gives you `agent-ab run` → p-value for agent pipelines.**

---

## 🗺️ Roadmap

- [x] Paired experiment execution
- [x] LLM-judge blind scoring
- [x] Welch's t-test with effect size
- [x] Bonferroni correction for multiple metrics
- [x] Bootstrap confidence intervals
- [x] Power analysis
- [x] CLI with Rich output
- [x] JSON export
- [x] CI/CD exit codes
- [ ] Bayesian analysis (posterior probability)
- [ ] Multi-armed bandit (sequential allocation)
- [ ] HTML shareable report
- [ ] Experiment registry (track all experiments)
- [ ] Automatic sample size calculation
- [ ] Stratified analysis by task type

---

## License

[MIT](LICENSE) — Test with confidence.

<div align="center">

**[agent-ab-tester](https://github.com/daniellopez882/agent-ab-tester)** by [Daniel López Orta](https://github.com/daniellopez882)

*Stop shipping on vibes. Ship on p-values.*

</div>

## Attribution
This project is based on [Ismail-2001/agent-ab-tester](https://github.com/Ismail-2001/agent-ab-tester), licensed under the MIT License.
Original author: Daniel Lopez Orta (daniellopezorta39@gmail.com); previously hosted at Ismail-2001/agent-ab-tester.
Modifications in this repository are by Sami Ahmed (sami.ahmed@ztech.com.pk).
The original LICENSE file and its copyright notice are preserved unchanged below.

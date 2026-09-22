"""Statistical analysis for A/B experiments — Frequentist and Bayesian."""

from __future__ import annotations

import math
import random
from typing import List
from dataclasses import dataclass

from ..models import MetricResult


@dataclass
class BayesianResult:
    """Result of a Bayesian A/B test."""
    p_better: float  # Probability that treatment is better than control
    expected_loss: float  # Expected relative loss if we choose treatment wrongly
    control_posterior_mean: float
    treatment_posterior_mean: float
    hdi_95_lower: float  # 95% High Density Interval
    hdi_95_upper: float


def welch_t_test(
    control_values: list[float],
    treatment_values: list[float],
    alpha: float = 0.05,
    name: str = "metric",
    direction: str = "higher_is_better",
    bonferroni_k: int = 1,
) -> MetricResult:
    """Perform Welch's t-test on paired differences."""
    n_c = len(control_values)
    n_t = len(treatment_values)

    if n_c < 2 or n_t < 2:
        return MetricResult(
            name=name,
            control_mean=_mean(control_values),
            treatment_mean=_mean(treatment_values),
            difference=0,
            relative_change=0,
            p_value=1.0,
            significant=False,
            direction=direction,
        )

    mean_c = _mean(control_values)
    mean_t = _mean(treatment_values)
    var_c = _variance(control_values)
    var_t = _variance(treatment_values)

    # Welch's t-statistic
    se = math.sqrt(var_c / n_c + var_t / n_t)
    if se == 0:
        return MetricResult(
            name=name,
            control_mean=mean_c,
            treatment_mean=mean_t,
            difference=mean_t - mean_c,
            relative_change=0,
            p_value=1.0,
            significant=False,
            direction=direction,
        )

    t_stat = (mean_t - mean_c) / se

    # Welch-Satterthwaite degrees of freedom
    num = (var_c / n_c + var_t / n_t) ** 2
    denom = (var_c / n_c) ** 2 / (n_c - 1) + (var_t / n_t) ** 2 / (n_t - 1)
    df = num / denom if denom > 0 else n_c + n_t - 2

    # Two-tailed p-value
    p_value = _t_distribution_p_value(abs(t_stat), df) * 2

    # Bonferroni correction
    adjusted_alpha = alpha / bonferroni_k

    # Effect size (Cohen's d)
    pooled_sd = math.sqrt(
        ((n_c - 1) * var_c + (n_t - 1) * var_t) / (n_c + n_t - 2)
    )
    cohens_d = (mean_t - mean_c) / pooled_sd if pooled_sd > 0 else 0

    # Confidence interval for the DIFFERENCE
    ci_margin = _t_critical(adjusted_alpha, df) * se
    ci_lower = (mean_t - mean_c) - ci_margin
    ci_upper = (mean_t - mean_c) + ci_margin

    # Relative change
    rel_change = (mean_t - mean_c) / mean_c if mean_c != 0 else 0

    # Determine which is better
    diff = mean_t - mean_c
    if direction == "higher_is_better":
        better = "treatment" if (diff > 0 and p_value < adjusted_alpha) else \
                 "control" if (diff < 0 and p_value < adjusted_alpha) else "neither"
    else:
        better = "treatment" if (diff < 0 and p_value < adjusted_alpha) else \
                 "control" if (diff > 0 and p_value < adjusted_alpha) else "neither"

    return MetricResult(
        name=name,
        control_mean=mean_c,
        treatment_mean=mean_t,
        difference=mean_t - mean_c,
        relative_change=rel_change,
        p_value=p_value,
        significant=p_value < adjusted_alpha,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        effect_size=cohens_d,
        effect_label=_effect_label(abs(cohens_d)),
        direction=direction,
        better_variant=better,
    )


def bayesian_analysis(
    control_values: List[float],
    treatment_values: List[float],
    samples: int = 10000,
) -> BayesianResult:
    """Estimate P(Treatment > Control) using a Normal-Normal conjugate model.
    Assuming scores are roughly normal or we have enough samples (CLT).
    """
    n_c, n_t = len(control_values), len(treatment_values)
    m_c, m_t = _mean(control_values), _mean(treatment_values)
    v_c, v_t = _variance(control_values), _variance(treatment_values)

    # Simple weak-prior Normal-Normal models
    # Simplified version: Monte Carlo sampling from the posterior distributions
    # Posterior for mean: Normal(mean=sample_mean, var=sample_var/n)
    
    # Pre-calculate std devs for sampling
    sd_c = math.sqrt(max(v_c, 0.001) / n_c)
    sd_t = math.sqrt(max(v_t, 0.001) / n_t)
    
    better_count = 0
    diffs = []
    for _ in range(samples):
        # Sample means from their posteriors
        c_mean_sample = random.gauss(m_c, sd_c)
        t_mean_sample = random.gauss(m_t, sd_t)
        
        if t_mean_sample > c_mean_sample:
            better_count += 1
        diffs.append(t_mean_sample - c_mean_sample)
    
    diffs.sort()
    hdi_lower = diffs[int(0.025 * samples)]
    hdi_upper = diffs[int(0.975 * samples)]
    
    return BayesianResult(
        p_better=better_count / samples,
        expected_loss=max(0, m_c - m_t), # Simple heuristic
        control_posterior_mean=m_c,
        treatment_posterior_mean=m_t,
        hdi_95_lower=hdi_lower,
        hdi_95_upper=hdi_upper
    )


def sequential_p_ratio_test(
    control_values: List[float],
    treatment_values: List[float],
    alpha: float = 0.05,
    beta: float = 0.10,
    mde: float = 0.1,  # Minimum Detectable Effect relative to mean
) -> str:
    """Simple SPRT implementation for continuous outcomes.
    Returns: 'continue', 'stop_success', 'stop_fail'
    """
    if len(control_values) < 20:
        return "continue"
    
    # Calculate log-likelihood ratio for a normal distribution
    # Simplified version using the cumulative difference and pooled variance
    m_c = _mean(control_values)
    v_pooled = (_variance(control_values) + _variance(treatment_values)) / 2
    sd_pooled = math.sqrt(max(v_pooled, 0.0001))
    
    # SPRT boundaries (Wald's)
    a = math.log((1 - beta) / alpha) # Upper limit (accept H1)
    b = math.log(beta / (1 - alpha)) # Lower limit (accept H0)
    
    # Cum Log Likelihood Ratio
    # Z = ln(L(H1)/L(H0))
    llh_ratio = 0
    mu_0 = m_c
    mu_1 = m_c * (1 + mde)
    
    for c, t in zip(control_values, treatment_values):
        # Difference observation
        diff = t - c
        # LLR for Normal: (x - mu0)^2/2s^2 - (x - mu1)^2/2s^2
        if sd_pooled > 0:
            term = (mu_1 - mu_0) * (diff - (mu_1 + mu_0) / 2) / (sd_pooled ** 2)
            llh_ratio += term
            
    if llh_ratio >= a:
        return "stop_success"
    if llh_ratio <= b:
        return "stop_fail"
    return "continue"


def bootstrap_ci(
    control_values: list[float],
    treatment_values: list[float],
    alpha: float = 0.05,
    name: str = "metric",
    direction: str = "higher_is_better",
    n_bootstrap: int = 10000,
) -> MetricResult:
    """Bootstrap confidence interval for the mean difference."""
    n = min(len(control_values), len(treatment_values))
    if n < 5:
        return welch_t_test(control_values, treatment_values, alpha, name, direction)

    observed_diff = _mean(treatment_values) - _mean(control_values)

    boot_diffs = []
    for _ in range(n_bootstrap):
        c_sample = [random.choice(control_values) for _ in range(n)]
        t_sample = [random.choice(treatment_values) for _ in range(n)]
        boot_diffs.append(_mean(t_sample) - _mean(c_sample))

    boot_diffs.sort()
    ci_lower = boot_diffs[int(alpha / 2 * n_bootstrap)]
    ci_upper = boot_diffs[int((1 - alpha / 2) * n_bootstrap)]

    count_null = sum(1 for d in boot_diffs if d <= 0)
    p_value = 2 * min(count_null, n_bootstrap - count_null) / n_bootstrap

    mean_c, mean_t = _mean(control_values), _mean(treatment_values)
    rel_change = (mean_t - mean_c) / mean_c if mean_c != 0 else 0

    significant = p_value < alpha
    if direction == "higher_is_better":
        better = "treatment" if (observed_diff > 0 and significant) else "control" if (observed_diff < 0 and significant) else "neither"
    else:
        better = "treatment" if (observed_diff < 0 and significant) else "control" if (observed_diff > 0 and significant) else "neither"

    return MetricResult(
        name=name,
        control_mean=mean_c,
        treatment_mean=mean_t,
        difference=observed_diff,
        relative_change=rel_change,
        p_value=max(p_value, 1 / n_bootstrap),
        significant=significant,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        direction=direction,
        better_variant=better,
    )


def power_analysis(
    effect_size: float = 0.5,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Estimate minimum sample size per group for desired power."""
    z_alpha = _z_score(alpha / 2)
    z_beta = _z_score(1 - power)
    n = math.ceil(2 * ((z_alpha + z_beta) / effect_size) ** 2)
    return max(n, 5)


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0

def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return sum((x - m) ** 2 for x in values) / (len(values) - 1)

def _effect_label(d: float) -> str:
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"

def _z_score(p: float) -> float:
    """Approximate z-score."""
    if p <= 0 or p >= 1:
        return 0.0
    if p > 0.5:
        return -_z_score(1 - p)
    t = math.sqrt(-2 * math.log(p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t)

def _t_critical(alpha: float, df: float) -> float:
    """Approximate t critical value."""
    if df > 120:
        return _z_score(alpha / 2)
    z = _z_score(alpha / 2)
    g1 = (z ** 3 + z) / (4 * df)
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)
    return z + g1 + g2

def _t_distribution_p_value(t_abs: float, df: float) -> float:
    """Approximate p-value from t-distribution."""
    if df > 120:
        return 0.5 * math.erfc(t_abs / math.sqrt(2))
    x = df / (df + t_abs ** 2)
    return max(min(0.5 * x ** (df / 2), 0.5), 0.0001)


def variance_reduction(
    y: list[float], 
    x: list[float], 
) -> list[float]:
    """CUPED Variance Reduction.
    Reduced variance = Var(Y) * (1 - Cor(Y,X)^2).
    """
    if len(y) != len(x) or len(y) < 2:
        return y
        
    mean_x = _mean(x)
    var_x = _variance(x)
    
    if var_x == 0:
        return y
        
    # Covariance
    mean_y = _mean(y)
    cov_yx = sum((yi - mean_y) * (xi - mean_x) for yi, xi in zip(y, x)) / (len(y) - 1)
    theta = cov_yx / var_x
    
    return [yi - theta * (xi - mean_x) for yi, xi in zip(y, x)]

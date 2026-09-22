"""CLI interface for agent-ab-tester."""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import sys
from pathlib import Path

import click
from rich.console import Console

from .experiment import Experiment
from .models import ExperimentReport
from .stats import power_analysis
from .reporters.dashboard import DashboardReporter
from .registry import ExperimentRegistry

console = Console()


def _load_variant(path: str):
    """Dynamically load an AgentVariant class from string 'module:Class'."""
    try:
        mod_name, class_name = path.split(":")
        module = importlib.import_module(mod_name)
        return getattr(module, class_name)()
    except Exception as e:
        click.echo(f"Error loading variant '{path}': {e}", err=True)
        sys.exit(1)


def _print_report(report: ExperimentReport):
    """Render the experiment report in the terminal with Rich."""
    from rich.table import Table
    from rich import box

    console.print(f"\n[bold blue]Experiment Results: {report.name}[/bold blue]")
    console.print(f"Confidence: {report.confidence:.0%}")
    console.print(f"Tasks: {report.num_tasks}")
    console.print(f"Elapsed: {report.elapsed_seconds:.1f}s")
    console.print(f"Total Cost: ${report.total_cost_usd:.4f}\n")

    table = Table(box=box.SIMPLE_HEAD)
    table.add_column("Metric")
    table.add_column("Control", justify="right")
    table.add_column("Treatment", justify="right")
    table.add_column("Difference", justify="right", style="bold")
    table.add_column("P-Value", justify="right")
    table.add_column("Sig", justify="center")

    for mr in report.metric_results:
        sig_marker = "✅" if mr.significant else "❌"
        table.add_row(
            mr.name,
            f"{mr.control_mean:.3f}",
            f"{mr.treatment_mean:.3f}",
            f"{mr.difference:+.3f} ({mr.relative_change:+.1%})",
            f"{mr.p_value:.4f}",
            sig_marker,
        )
    console.print(table)
    
    # Bayesian summary
    console.print(f"\n[bold]Probability Treatment > Control: {report.bayesian_p_better:.1%}[/bold]")
    
    color = "green" if "Ship" in report.recommendation else "yellow"
    console.print(f"\n[bold {color}]Verdict: {report.recommendation}[/bold {color}]\n")


@click.group()
def cli():
    """Agent A/B Tester — Enterprise grade experimentation for AI pipelines."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")


@cli.command()
@click.option("--control", required=True, help="Control agent as 'module:Class'")
@click.option("--treatment", required=True, help="Treatment agent as 'module:Class'")
@click.option("--tasks", required=True, help="Tasks JSONL file")
@click.option("--num-tasks", type=int, help="Number of tasks to run")
@click.option("--confidence", type=float, default=0.95, help="Confidence level (0-1)")
@click.option("--method", type=click.Choice(["welch_t", "bootstrap"]), default="welch_t")
@click.option("--judge-model", default="gpt-4o-mini")
@click.option("--name", default="experiment", help="Experiment name")
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table")
@click.option("-o", "--output", type=click.Path())
@click.option("--dashboard", type=click.Path(), help="Path to save HTML dashboard")
@click.option("--require-significant", help="Fail if metric not significant for treatment")
@click.option("--exit-code", is_flag=True)
@click.option("--red-team", is_flag=True, help="Auto-generate adversarial tasks for stress testing")
def run(control, treatment, tasks, num_tasks, confidence, method,
        judge_model, name, fmt, output, dashboard, require_significant, exit_code, red_team):
    """Run an A/B experiment between two agent variants."""
    control_agent = _load_variant(control)
    treatment_agent = _load_variant(treatment)

    experiment = Experiment(
        control=control_agent,
        treatment=treatment_agent,
        name=name,
        num_tasks=num_tasks,
        confidence=confidence,
        method=method,
        judge_model=judge_model,
    )

    if red_team:
        click.echo("🛡️  Applying Adversarial Red-Teaming to your task set...")
        # We manually load tasks to feed into stress_test
        raw_tasks = asyncio.run(experiment.load_tasks(tasks))
        report = asyncio.run(experiment.stress_test(raw_tasks, num_adversarial=num_tasks or 20))
    else:
        report = asyncio.run(experiment.run_from_file(tasks))

    if fmt == "json":
        json_str = json.dumps(report.to_dict(), indent=2)
        if output:
            Path(output).write_text(json_str)
            click.echo(f"Report saved → {output}")
        else:
            click.echo(json_str)
    else:
        _print_report(report)
        if output:
            Path(output).write_text(json.dumps(report.to_dict(), indent=2))
            
    if dashboard:
        dr = DashboardReporter(dashboard)
        dr.generate(report)
        click.echo(f"Dashboard generated → {dashboard}")
        
    # Registry
    reg = ExperimentRegistry()
    exp_id = reg.add_report(report)
    click.echo(f"Experiment saved to history (ID: {exp_id})")

    # CI check
    if require_significant and exit_code:
        is_sig = any(m.name == require_significant and m.significant for m in report.metric_results)
        if not is_sig:
            click.echo(f"Error: Metric '{require_significant}' was not significant.", err=True)
            sys.exit(1)


@cli.command()
@click.option("--effect-size", type=float, default=0.5)
@click.option("--alpha", type=float, default=0.05)
@click.option("--power", type=float, default=0.8)
def plan(effect_size, alpha, power):
    """Plan an experiment by calculating required sample size."""
    n = power_analysis(effect_size, alpha, power)
    click.echo(f"\n[bold green]Experiment Plan[/bold green]")
    click.echo(f"Target Effect Size (Cohen's d): {effect_size}")
    click.echo(f"Power (1-Beta): {power}")
    click.echo(f"Alpha (Significance): {alpha}")
    click.echo("-" * 20)
    click.echo(f"Recommended samples per group: {n}")
    click.echo(f"  → Total tasks needed: {n * 2}")
    click.echo()


@cli.command()
@click.option("--domain", required=True, help="Description of the domain (e.g. 'coding assistant')")
@click.option("--num-tasks", type=int, default=20, help="Number of tasks to generate")
@click.option("--output", type=click.Path(), default="synthetic_tasks.jsonl")
def synthesize(domain, num_tasks, output):
    """Synthesize a diverse set of tasks for a given domain."""
    from .synthesis import TaskSynthesizer
    import asyncio
    
    click.echo(f"✨  Generating {num_tasks} synthetic tasks for '{domain}'...")
    
    synth = TaskSynthesizer()
    tasks = asyncio.run(synth.generate_tasks(domain, num_tasks))
    
    with open(output, "w") as f:
        for t in tasks:
            f.write(json.dumps({"input": t}) + "\n")
            
    click.echo(f"✅  Synthetic dataset saved to {output}")


@cli.command()
@click.option("--limit", type=int, default=10)
def history(limit):
    """View historical experiment runs."""
    reg = ExperimentRegistry()
    exps = reg.list_experiments(limit)
    
    if not exps:
        click.echo("No experiment history found.")
        return
        
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
        
        console = Console()
        table = Table(title="Experiment History", box=box.ROUNDED)
        table.add_column("ID", style="dim")
        table.add_column("Name", style="bold")
        table.add_column("Timestamp", style="cyan")
        table.add_column("Tasks", justify="right")
        table.add_column("P(Superiority)", justify="right")
        table.add_column("Verdict")
        
        for e in exps:
            p_val = f"{e['bayesian_p_better']:.1%}" if e['bayesian_p_better'] else "N/A"
            table.add_row(
                str(e['id']), e['name'], e['timestamp'], 
                str(e['num_tasks']), p_val, e['recommendation']
            )
        console.print(table)
    except ImportError:
        for e in exps:
            click.echo(f"{e['id']:>3} │ {e['name']:<20} │ {e['timestamp']} │ {e['num_tasks']:>3} tasks │ {e['recommendation']}")


def main():
    cli()


if __name__ == "__main__":
    main()

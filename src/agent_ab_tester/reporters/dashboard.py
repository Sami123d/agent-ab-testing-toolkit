"""Dashboard reporter — generates a high-fidelity HTML report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import ExperimentReport

class DashboardReporter:
    """Generates a modern, interactive HTML dashboard for experiment results."""

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)

    def generate(self, report: ExperimentReport) -> Path:
        """Create the HTML file with embedded data."""
        data = report.to_dict()
        # Add extra details for the dashboard
        data["control_name"] = report.control_name
        data["treatment_name"] = report.treatment_name
        
        html = self._get_template(data)
        self.output_path.write_text(html, encoding="utf-8")
        return self.output_path

    def _get_template(self, data: dict) -> str:
        data_json = json.dumps(data)
        
        return f"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>agent-ab-tester | {data['experiment']}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://unpkg.com/lucide@latest"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <script>
        tailwind.config = {{
            darkMode: 'class',
            theme: {{
                extend: {{
                    fontFamily: {{
                        sans: ['Inter', 'sans-serif'],
                    }},
                    colors: {{
                        brand: {{
                            50: '#f0f9ff',
                            100: '#e0f2fe',
                            500: '#0ea5e9',
                            600: '#0284c7',
                            900: '#0c4a6e',
                        }}
                    }}
                }}
            }}
        }}
    </script>
    <style>
        body {{ font-family: 'Inter', sans-serif; }}
        .glass {{
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .score-card {{
            transition: all 0.3s ease;
        }}
        .score-card:hover {{
            transform: translateY(-2px);
            border-color: rgba(14, 165, 233, 0.4);
        }}
    </style>
</head>
<body class="bg-[#0b0f19] text-slate-100 min-h-screen">
    <div class="max-w-7xl mx-auto px-4 py-12">
        <!-- Header -->
        <header class="flex flex-col md:flex-row md:items-center justify-between mb-12 gap-6">
            <div>
                <div class="flex items-center gap-3 mb-2">
                    <span class="p-2 bg-brand-500/20 text-brand-500 rounded-lg">
                        <i data-lucide="beaker" class="w-6 h-6"></i>
                    </span>
                    <h1 class="text-3xl font-bold tracking-tight">Experiment: <span class="text-brand-500">{data['experiment']}</span></h1>
                </div>
                <p class="text-slate-400">Benchmarking <span class="text-slate-200 font-medium">{data['control_name']}</span> vs <span class="text-slate-200 font-medium">{data['treatment_name']}</span></p>
            </div>
            <div class="flex items-center gap-4">
                <div class="text-right hidden md:block">
                    <p class="text-sm text-slate-500 uppercase tracking-widest font-semibold">Bayesian Prob.</p>
                    <p class="text-2xl font-bold text-emerald-400">{data['bayesian_p_better']:.1%} Superiority</p>
                </div>
                <div class="h-12 w-px bg-slate-800 ml-2"></div>
                <div class="p-4 glass rounded-2xl flex items-center gap-3">
                    <div class="text-right">
                        <p class="text-xs text-slate-500 font-semibold uppercase">Verdict</p>
                        <p class="font-bold text-white uppercase tracking-wide">{data['recommendation']}</p>
                    </div>
                </div>
            </div>
        </header>

        <!-- Stats Grid -->
        <section class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-12">
            <div class="glass p-6 rounded-2xl">
                <div class="flex items-center justify-between mb-4">
                    <p class="text-slate-400 text-sm font-medium">Total Tasks</p>
                    <i data-lucide="check-circle" class="w-4 h-4 text-brand-500"></i>
                </div>
                <p class="text-3xl font-bold">{data['num_tasks']}</p>
            </div>
            <div class="glass p-6 rounded-2xl">
                <div class="flex items-center justify-between mb-4">
                    <p class="text-slate-400 text-sm font-medium">Experiment Cost</p>
                    <i data-lucide="dollar-sign" class="w-4 h-4 text-brand-500"></i>
                </div>
                <p class="text-3xl font-bold">${data['total_cost_usd']:.2f}</p>
            </div>
            <div class="glass p-6 rounded-2xl">
                <div class="flex items-center justify-between mb-4">
                    <p class="text-slate-400 text-sm font-medium">Confidence Level</p>
                    <i data-lucide="shield-check" class="w-4 h-4 text-emerald-500"></i>
                </div>
                <p class="text-3xl font-bold">{data['confidence']*100:.0f}%</p>
            </div>
            <div class="glass p-6 rounded-2xl">
                <div class="flex items-center justify-between mb-4" >
                    <p class="text-slate-400 text-sm font-medium">Duration</p>
                    <i data-lucide="clock" class="w-4 h-4 text-brand-500"></i>
                </div>
                <p class="text-3xl font-bold">{data['elapsed_seconds']:.1f}s</p>
            </div>
        </section>

        <!-- Main Metrics -->
        <h2 class="text-xl font-semibold mb-6 flex items-center gap-2">
            <i data-lucide="bar-chart-3" class="w-5 h-5"></i> Key Performance Indicators
        </h2>
        
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
            <!-- Metric Cards -->
            <div class="space-y-4" id="metric-cards">
                <!-- Injected by JS -->
            </div>
            
            <!-- Visualization -->
            <div class="glass p-8 rounded-3xl relative overflow-hidden">
                <div class="absolute top-0 right-0 p-8 opacity-10">
                    <i data-lucide="activity" class="w-32 h-32"></i>
                </div>
                <h3 class="text-lg font-medium mb-6">Distribution Comparison</h3>
                <div class="h-64">
                    <canvas id="mainChart"></canvas>
                </div>
            </div>
        </div>

        <!-- Detail Table -->
        <section>
            <h2 class="text-xl font-semibold mb-6 flex items-center gap-2">
                <i data-lucide="list" class="w-5 h-5"></i> Detailed Statistical Report
            </h2>
            <div class="glass rounded-3xl overflow-hidden">
                <table class="w-full text-left">
                    <thead>
                        <tr class="bg-slate-900 border-b border-slate-800">
                            <th class="px-6 py-4 text-sm font-semibold text-slate-300">METRIC</th>
                            <th class="px-6 py-4 text-sm font-semibold text-slate-300 text-right">{data['control_name']}</th>
                            <th class="px-6 py-4 text-sm font-semibold text-slate-300 text-right">{data['treatment_name']}</th>
                            <th class="px-6 py-4 text-sm font-semibold text-slate-300 text-right">LIFT</th>
                            <th class="px-6 py-4 text-sm font-semibold text-slate-300 text-right">P-VALUE</th>
                            <th class="px-6 py-4 text-sm font-semibold text-slate-300 text-right">SIGNIFICANT</th>
                        </tr>
                    </thead>
                    <tbody id="table-body">
                        <!-- Injected by JS -->
                    </tbody>
                </table>
            </div>
        </section>

        <footer class="mt-20 py-8 border-t border-slate-900 text-center text-slate-500 text-sm">
            <p>Generated by <strong>agent-ab-tester</strong> v2.0 &middot; Statistical analysis for AI professionals.</p>
        </footer>
    </div>

    <script>
        const data = {data_json};
        
        function init() {{
            lucide.createIcons();
            renderMetrics();
            renderChart();
        }}

        function renderMetrics() {{
            const container = document.getElementById('metric-cards');
            const tbody = document.getElementById('table-body');
            
            data.metrics.forEach(m => {{
                // Card
                const lift = m.relative_change;
                const liftColor = lift > 0 ? (m.name === 'quality' ? 'text-emerald-400' : 'text-rose-400') : 
                                  lift < 0 ? (m.name === 'quality' ? 'text-rose-400' : 'text-emerald-400') : 'text-slate-400';
                const liftIcon = lift > 0 ? 'arrow-up-right' : lift < 0 ? 'arrow-down-right' : 'minus';
                
                const card = document.createElement('div');
                card.className = "glass p-6 rounded-2xl score-card flex items-center justify-between";
                card.innerHTML = `
                    <div>
                        <p class="text-sm font-medium text-slate-500 uppercase tracking-wider mb-1">${{m.name}}</p>
                        <div class="flex items-baseline gap-3">
                            <h4 class="text-2xl font-bold">${{m.treatment_mean.toFixed(2)}}</h4>
                            <span class="text-sm text-slate-500">vs ${{m.control_mean.toFixed(2)}}</span>
                        </div>
                    </div>
                    <div class="text-right">
                        <div class="flex items-center gap-1 ${{liftColor}} font-bold">
                            <i data-lucide="${{liftIcon}}" class="w-4 h-4"></i>
                            <span>${{(lift * 100).toFixed(1)}}%</span>
                        </div>
                        <p class="text-xs text-slate-500">$\pm${{((m.ci_upper - m.ci_lower)/2).toFixed(3)}} interval</p>
                    </div>
                `;
                container.appendChild(card);

                // Table
                const row = document.createElement('tr');
                row.className = "border-b border-slate-800/50 hover:bg-white/5 transition-colors";
                row.innerHTML = `
                    <td class="px-6 py-4 font-semibold text-slate-200 uppercase text-xs tracking-widest">${{m.name}}</td>
                    <td class="px-6 py-4 text-right transform tabular-nums">${{m.control_mean.toFixed(4)}}</td>
                    <td class="px-6 py-4 text-right font-bold transform tabular-nums">${{m.treatment_mean.toFixed(4)}}</td>
                    <td class="px-6 py-4 text-right ${{liftColor}} font-bold transform tabular-nums">${{(lift * 100).toFixed(1)}}%</td>
                    <td class="px-6 py-4 text-right text-slate-400 tabular-nums">${{m.p_value.toFixed(4)}}</td>
                    <td class="px-6 py-4 text-right">
                        <span class="px-2 py-1 rounded-full text-[10px] uppercase font-bold tracking-tighter ${{m.significant ? 'bg-emerald-500/20 text-emerald-400' : 'bg-slate-800 text-slate-500'}}">
                            ${{m.significant ? 'SIGNIFICANT' : 'STOCHASTIC'}}
                        </span>
                    </td>
                `;
                tbody.appendChild(row);
            }});
            lucide.createIcons();
        }}

        function renderChart() {{
            const ctx = document.getElementById('mainChart').getContext('2d');
            const qualityMetric = data.metrics.find(m => m.name === 'quality');
            
            new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: [data.control_name, data.treatment_name],
                    datasets: [{{
                        label: 'Quality Score',
                        data: [qualityMetric.control_mean, qualityMetric.treatment_mean],
                        backgroundColor: ['rgba(148, 163, 184, 0.2)', 'rgba(14, 165, 233, 0.4)'],
                        borderColor: ['rgba(148, 163, 184, 1)', 'rgba(14, 165, 233, 1)'],
                        borderWidth: 2,
                        borderRadius: 12,
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            max: 10,
                            grid: {{ color: 'rgba(255, 255, 255, 0.05)' }},
                            ticks: {{ color: '#94a3b8' }}
                        }},
                        x: {{
                            grid: {{ display: false }},
                            ticks: {{ color: '#94a3b8' }}
                        }}
                    }},
                    plugins: {{
                        legend: {{ display: false }}
                    }}
                }}
            }});
        }}

        init();
    </script>
</body>
</html>"""

if __name__ == "__main__":
    # Test stub
    pass

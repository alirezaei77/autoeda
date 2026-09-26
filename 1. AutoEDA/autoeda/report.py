"""
HTML report builder using Jinja2 templates.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from jinja2 import Template

from .analyzer import AutoEDAReport

_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AutoEDA Report - {{ report.dataset_name }}</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         margin: 0; padding: 0; background: #f7f7f9; color: #222; }
  header { background: #2c3e50; color: #fff; padding: 24px 40px; }
  header h1 { margin: 0; font-size: 24px; }
  header p { margin: 4px 0 0; opacity: .85; }
  main { max-width: 1200px; margin: 0 auto; padding: 24px 40px 60px; }
  section { background: #fff; border-radius: 8px; padding: 20px 24px;
            margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
  h2 { border-bottom: 2px solid #eee; padding-bottom: 8px; margin-top: 0; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; }
  th, td { border: 1px solid #e0e0e0; padding: 6px 10px; text-align: left; }
  th { background: #f0f3f7; }
  tr:nth-child(even) { background: #fafbfc; }
  img { max-width: 100%; border-radius: 6px; margin: 8px 0; }
  .badge { display: inline-block; background: #3498db; color: #fff;
           border-radius: 12px; padding: 2px 10px; font-size: 12px; margin-left: 6px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 800px) { .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <h1>AutoEDA Report <span class="badge">{{ report.dataset_name }}</span></h1>
  <p>Shape: {{ report.shape[0] }} rows × {{ report.shape[1] }} columns &nbsp;|&nbsp;
     Column coverage: {{ report.coverage_pct }}%</p>
</header>
<main>

<section>
  <h2>Problem Detection</h2>
  <p><strong>Type:</strong> {{ report.problem_info.problem_type }}</p>
  <p><strong>Target:</strong> {{ report.problem_info.target or "—" }}</p>
  <p><strong>Rationale:</strong> {{ report.problem_info.rationale }}</p>
</section>

{% if report.numeric_summary is not none and not report.numeric_summary.empty %}
<section>
  <h2>Numeric Summary</h2>
  {{ report.numeric_summary.to_html(classes="table", border=0) | safe }}
</section>
{% endif %}

{% if report.categorical_summary %}
<section>
  <h2>Categorical Summary</h2>
  {% for col, tbl in report.categorical_summary.items() %}
    <h3>{{ col }}</h3>
    {{ tbl.to_html(classes="table", border=0) | safe }}
  {% endfor %}
</section>
{% endif %}

{% if not report.missing_summary.empty %}
<section>
  <h2>Missing Values</h2>
  {{ report.missing_summary.to_html(classes="table", border=0) | safe }}
</section>
{% endif %}

{% if not report.outlier_iqr.empty %}
<section>
  <h2>Outliers (IQR)</h2>
  {{ report.outlier_iqr.to_html(classes="table", border=0) | safe }}
</section>
{% endif %}

{% if not report.outlier_z.empty %}
<section>
  <h2>Outliers (Z-score)</h2>
  {{ report.outlier_z.to_html(classes="table", border=0) | safe }}
</section>
{% endif %}

{% if report.correlations %}
<section>
  <h2>Correlations</h2>
  <div class="grid">
  {% for name, mat in report.correlations.items() %}
    <div>
      <h3>{{ name|capitalize }} correlation</h3>
      {{ mat.to_html(classes="table", border=0) | safe }}
    </div>
  {% endfor %}
  </div>
</section>
{% endif %}

{% if not report.normality.empty %}
<section>
  <h2>Normality Tests</h2>
  {{ report.normality.to_html(classes="table", border=0) | safe }}
</section>
{% endif %}

{% if report.plots %}
<section>
  <h2>Plots</h2>
  {% for p in report.plots %}
    <div><img src="plots/{{ p }}" alt="{{ p }}"></div>
  {% endfor %}
</section>
{% endif %}

</main>
</body>
</html>
"""


def build_html_report(report: AutoEDAReport, out_path: Path, base_dir: Path) -> None:
    """Render the HTML report to `out_path`."""
    html = Template(_TEMPLATE).render(report=report)
    out_path.write_text(html, encoding="utf-8")
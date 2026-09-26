\# AutoEDA — Automated Exploratory Data Analysis



A lightweight, dependency-friendly Python toolkit that generates a \*\*full EDA

report (HTML + plots + tables)\*\* for any tabular dataset, with \*\*CLI\*\*,

\*\*problem-type detection\*\*, \*\*multi-dataset comparison\*\*, and \*\*unit tests\*\*.



\## Features



\- Descriptive statistics for numeric \& categorical columns

\- Missing-value analysis (matrix, bar, summary table via `missingno`)

\- Outlier detection using \*\*IQR\*\* and \*\*Z-score\*\*

\- Correlation matrices (\*\*Pearson\*\* \& \*\*Spearman\*\*)

\- Normality tests (\*\*Shapiro-Wilk\*\*, \*\*Anderson-Darling\*\*)

\- Auto-detection of ML problem type (binary/multiclass/regression/unsupervised)

\- Rich \*\*HTML report\*\* with all tables and figures

\- \*\*JSON\*\* summary export

\- \*\*CLI\*\* for single \& multi-dataset runs

\- \*\*Unit tests\*\* with `pytest`



\## Install



```bash

pip install -r requirements.txt

pip install -e .

```



\## Usage — Python API



```python

import pandas as pd

from autoeda import AutoEDA



df = pd.read\_csv("data/titanic.csv")

eda = AutoEDA(df, dataset\_name="titanic", target="Survived")

report = eda.run("reports/titanic")



print(report.problem\_info)

print(f"Coverage: {report.coverage\_pct}%")

```



\## Usage — CLI



```bash

\# Single dataset

python -m autoeda.cli --builtin iris --target target --output reports/iris



\# Compare all builtin datasets

python -m autoeda.cli --compare --output reports/all



\# From CSV

python -m autoeda.cli --input data/wine.csv --target quality --output reports/wine

```



\## Output Structure



```

reports/iris/

├── report.html          # Full HTML report

├── report.json          # Machine-readable summary

└── plots/

&#x20;   ├── histograms.png

&#x20;   ├── boxplots.png

&#x20;   ├── corr\_pearson.png

&#x20;   ├── corr\_spearman.png

&#x20;   ├── categorical\_bars.png

&#x20;   ├── missing\_matrix.png

&#x20;   └── target\_distribution.png

```



\## Tests



```bash

pytest -q

```



\## Success Criteria



\- Runs without error on \*\*Iris\*\*, \*\*Wine\*\*, \*\*Titanic\*\*, \*\*California Housing\*\*

\- Column coverage ≥ \*\*90%\*\*

\- Zero hard failures in unit tests



\## Roadmap



\- \[x] CLI

\- \[x] Auto tests

\- \[x] Multi-dataset comparison

\- \[x] Problem-type detection

\- \[ ] PDF export (WeasyPrint)

\- \[ ] Interactive Plotly dashboard

\- \[ ] Skew-aware feature recommendations



\## License



MIT


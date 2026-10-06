# Freight Rate Prediction

CatBoost models that predict freight load rates. The models are trained on loads from January to October 2025 and tested on September to October.

## Files

| File | Purpose |
|---|---|
| `train.py` | Trains the full model, prints test metrics, writes `validation_predictions.csv` |
| `dec_eval.py` | Trains the December model (only the inputs the December file has), writes `data/december_chart_inputs.csv` |
| `score.py` | Provided scorer: validates both outputs and creates the December chart |


Input data files are in the project root: `train-test.csv`, `validation.csv`, `december-chart-inputs.csv`.

## Run

```bash
python -m pip install -r requirements.txt

python train.py      # writes validation_predictions.csv
python dec_eval.py   # writes data/december_chart_inputs.csv

python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

The scorer creates `scorer_results/candidate_december.png`.


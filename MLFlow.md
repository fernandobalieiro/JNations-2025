# MLflow UI Walkthrough: How to Inspect the Lifecycle

Once you run `train.py` and MLflow has logged your experiments, you can inspect the full model lifecycle using MLflow's UI:

## 🚀 Launch the MLflow UI

```bash
mlflow ui
```

Then open your browser to:
```
http://localhost:5000
```

## 🔍 What to Look For

### 1. **Experiments tab**
- You'll see: `adult-income-workshop`
- Click into it to view all runs

### 2. **Compare Runs**
- Check `accuracy`, `f1_score`, `max_iter`, `max_depth`
- Sort by F1 score to confirm which model was best

### 3. **Model Registry (Models tab)**
- Look for:
  - `LogisticRegression-AdultIncome-Model`
  - `DecisionTree-AdultIncome-Model`
- Click to view model versions and staging history

### 4. **Artifacts**
- For each run, expand the "Artifacts" section to download:
  - model.pkl
  - sklearn model files
  - metrics.json (auto-logged by MLflow)

## 🧠 Bonus: Production Monitoring
- If your model has been promoted to `Production`, you'll see its version tagged.
- This helps track what’s running live.

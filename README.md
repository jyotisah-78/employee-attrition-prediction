# Employee Attrition Prediction

Predicts whether an employee will leave (1) or stay (0) using two public Kaggle datasets
(IBM HR Analytics, 1,470 rows + Employee HR, ~15,000 rows) and several classifiers
(RandomForest, HistGradientBoosting, XGBoost, LightGBM, CatBoost, and a stacked ensemble).

This is a **command-line machine-learning project** (Python). It has no website or database.

## Project structure

```
employee-attrition-prediction/
├── main.py                      <- the file you run
├── requirements.txt             <- libraries to install
├── .env.example                 <- template for Kaggle login (copy to .env)
├── .gitignore
├── README.md
├── src/attrition/
│   ├── config.py                <- all settings (paths, seeds, model sizes)
│   ├── data.py                  <- loads CSVs, makes both datasets share one format
│   ├── features.py              <- new features + the target-leakage guard
│   ├── modeling.py              <- preprocessing, SMOTE, model definitions
│   ├── evaluation.py            <- accuracy/AUC/F1/recall, cross-validation, baseline
│   ├── plots.py                 <- comparison chart, ROC, confusion matrix, CV folds
│   ├── pipeline.py              <- runs everything in order, saves outputs
│   └── synthetic.py             <- FAKE data generator (for testing only)
├── scripts/
│   ├── download_data.py         <- downloads both datasets from Kaggle
│   └── make_synthetic_data.py   <- writes FAKE demo CSVs
├── tests/                       <- automatic checks
├── notebooks/original_employee_attrition_pipeline.ipynb   <- your original, untouched
├── data/raw/                    <- put the CSV files here
├── results/                     <- tables and charts appear here
└── models/                      <- the saved best model appears here
```

## Setup on a MacBook (do this once)

1. Install Homebrew (if you don't have it): https://brew.sh
2. In Terminal:
   ```bash
   brew install python@3.12 libomp
   ```
   `libomp` is required: without it XGBoost and LightGBM fail to import on macOS.
3. In VS Code: **File → Open Folder… → select `employee-attrition-prediction`** (the folder that contains `main.py`).
4. Open VS Code's terminal (**Terminal → New Terminal**) and run:
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   Your prompt should now start with `(.venv)`. Re-run `source .venv/bin/activate` every time you open a new terminal.
5. In VS Code press **Cmd+Shift+P → "Python: Select Interpreter" → choose `.venv`**.

## Get the data (pick ONE way)

**A. Kaggle download (real data)**
1. kaggle.com → profile picture → Settings → API → **Create New Token** (downloads `kaggle.json`).
2. Either `mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`,
   or `cp .env.example .env` and fill in the username and key.
3. Open each dataset page once in the browser and accept its terms if asked, then:
   ```bash
   python scripts/download_data.py
   ```

**B. Manual**: download both datasets from Kaggle's website, unzip, and put
`WA_Fn-UseC_-HR-Employee-Attrition.csv` and `Employee_HR.csv` inside `data/raw/`.

**C. Fake data (only to check the code runs):** `python scripts/make_synthetic_data.py`
The numbers from fake data mean nothing. Never show them as results.

## Run

```bash
python main.py --quick            # fast first run (fewer trees, 3-fold CV)
python main.py                    # full run, both datasets merged
python main.py --mode ibm         # IBM dataset only
python main.py --mode employee    # Employee HR dataset only
python main.py --models RandomForest,HistGB   # only some models
python main.py --help
```

Run all three modes before presenting. See "Honest limitations" for why.

## Outputs

In `results/`: `<mode>_results.csv`, `<mode>_summary.json`, `<mode>_model_comparison.png`,
`<mode>_roc_curves.png`, `<mode>_confusion_matrix.png`, `<mode>_cv_folds.png`.
In `models/`: `<mode>_best_model.joblib` (the whole fitted pipeline).

## Run the tests

```bash
python -m unittest discover -s tests -t . -v
```
Expected: `Ran 7 tests ... OK` (takes about 10-30 seconds; uses fake data in a temporary folder).

## What was fixed compared with the original notebook

| Problem in the original notebook | Fix |
|---|---|
| Chart in section 4 used typed-in numbers (e.g. CatBoost 0.883) that were never computed | All charts now use real results from the run |
| `Config` lacked many fields (`exclude_cols`, `test_size`, `cv_folds`, ...), causing a crash | One complete `Config` in `config.py` |
| Placeholder cell `la alaa` instead of ROC curves | Real ROC plot |
| Only the first label was dropped; `Churn` / `left` could stay in the features | Label columns removed in the loaders, plus a guard that stops the program if one slips through |
| Imputation, scaling and encoding ran on all data before the split | All learning steps are inside a scikit-learn Pipeline, fitted on training rows only |
| Label encoding made up of fake ordering and SMOTE on integer codes | `OrdinalEncoder` + `SMOTENC` (knows which columns are categories) |
| Best model chosen using the test set | Chosen by cross-validation on the training set; test set used once |
| CatBoost and Stacking silently did not run | Optional libraries are detected and reported; stack built from whatever is installed |
| CUDA/GPU detection (no GPU on a Mac) | Removed; CPU only |
| `os.system("kaggle ...")` ignored all errors | Download script checks and reports errors |

## Honest limitations (read before presenting)

1. **The "80% accuracy" goal is weak.** About 77% of employees stay, so a "model" that always says
   "stays" scores about 77%. Judge models by ROC-AUC, balanced accuracy and **recall on leavers**
   (how many leavers are actually caught). The program prints the baseline for you.
2. **The two datasets share only a few columns** (satisfaction, income, tenure, department).
   After merging, many columns are empty for one source and filled with the median/"Unknown".
   The two sources also have different leave rates, so the model can partly learn "which dataset is this row from".
   Comparing `--mode merged` with `--mode ibm` and `--mode employee` shows how much this matters.
3. **Very high accuracy (like 98-99%) needs an explanation.** The Employee HR data is known to be
   easy for tree models, so a high score may be partly real, but it can also mean leakage. The program
   warns above 97%. Be ready to explain which of the two it is for your run.
4. **Assumptions to verify against your real CSVs** (marked in `data.py`): the satisfaction scale of the
   Employee HR file is guessed from its maximum value, `Salary_INR` is treated as a yearly amount,
   and 1 USD = 75 INR (`config.py`).
5. SMOTE adds synthetic rows to the training data only. It often does not improve ROC-AUC for tree
   models; try `--no-smote` and compare.

## Troubleshooting

| Message | Fix |
|---|---|
| `Library not loaded: libomp.dylib` | `brew install libomp`, then restart the terminal |
| `kaggle: command not found` | Activate the venv: `source .venv/bin/activate` |
| `403 Forbidden` from Kaggle | Open the dataset page in the browser and accept its terms |
| `Could not find '...csv'` | Put the CSVs in `data/raw/` (see "Get the data") |
| `No module named attrition` | Run commands from the project root folder, not from inside `src/` |

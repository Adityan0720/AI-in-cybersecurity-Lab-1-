# AI-Based Cyber Threat Detection

An academic mini project for the AI in Cyber Security Lab. It covers dataset analysis, basic machine-learning threat classification, a simple neural-network prototype, and model comparison.

Task 5, ethical reflection and AI challenges, is intentionally excluded.

## Technologies

- Python
- pandas and NumPy
- scikit-learn
- Matplotlib and Seaborn
- Jupyter Notebook

## Structure

- `data/`: Dataset files and source documentation
- `notebooks/`: The complete lab notebook
- `src/`: Reusable loading, preprocessing, EDA, model, and comparison modules
- `results/plots/`: Generated visualizations
- `results/models/`: Saved trained models
- `results/metrics/`: Saved evaluation metrics
- `report/`: Report template and documentation

## Installation

Create and activate a virtual environment, then install the required packages:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Running the Project

Open `notebooks/cybersecurity_analysis.ipynb` in VS Code or Jupyter and run the sections in order after confirming the dataset format and target column.

The reusable modules in `src/` are intentionally framework-only at this stage. Model execution and result generation must use the actual dataset and must not use invented metric values.

## Expected Outputs

After the notebook is completed and run, generated plots will be saved in `results/plots/`, trained models in `results/models/`, and metric tables in `results/metrics/`.

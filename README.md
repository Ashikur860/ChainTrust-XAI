# ChainTrust-XAI

## Explainable Machine Learning Framework for Illicit Bitcoin Transaction Detection Using Ensemble Learning

ChainTrust-XAI is a machine learning-based framework designed to detect illicit Bitcoin transactions using the Elliptic Bitcoin Dataset. The project evaluates multiple machine learning algorithms and proposes a hybrid ensemble approach to improve blockchain transaction risk detection.

The framework compares six different machine learning models and generates publication-ready evaluation metrics and visualizations suitable for academic research.

---

## Features

- Bitcoin transaction classification
- Data preprocessing and label mapping
- Six machine learning models
- Hybrid Ensemble Learning
- Performance comparison
- Confusion Matrix generation
- ROC Curve generation
- Feature Importance analysis
- Publication-ready figures

---

## Machine Learning Models

The following models are implemented:

- Logistic Regression
- Decision Tree
- Support Vector Machine (SVM)
- Random Forest
- XGBoost
- Hybrid Ensemble (Random Forest + XGBoost Soft Voting)

---

## Dataset

This project uses the **Elliptic Bitcoin Dataset**, one of the most widely used public datasets for illicit cryptocurrency transaction detection.

Dataset contains:

- Bitcoin Transactions
- Transaction Labels
- Blockchain Transaction Graph

Files required:

```
elliptic_txs_features.csv
elliptic_txs_classes.csv
elliptic_txs_edgelist.csv
```

---

## Project Structure

```
ChainTrust-XAI/
│
├── main.py
├── requirements.txt
│
├── data/
│   ├── elliptic_txs_features.csv
│   ├── elliptic_txs_classes.csv
│   └── elliptic_txs_edgelist.csv
│
├── figures/
│   ├── xgboost_confusion_matrix.png
│   ├── ensemble_confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
│
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/your-username/ChainTrust-XAI.git
```

Move into the project

```bash
cd ChainTrust-XAI
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run

```bash
python main.py
```

The program will automatically:

- Load dataset
- Train all models
- Compare performance
- Generate evaluation metrics
- Save publication-quality figures

---

## Output

Generated figures include:

- XGBoost Confusion Matrix
- Ensemble Confusion Matrix
- ROC Curve
- Top-10 Feature Importance

Model comparison table is also displayed in the terminal.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn

---

## Research Objective

The objective of this project is to develop an explainable machine learning framework capable of accurately identifying illicit Bitcoin transactions while comparing multiple classification algorithms and evaluating the effectiveness of ensemble learning for blockchain analytics.

---

## Future Work

Future improvements may include:

- SHAP Explainability
- Graph Neural Networks (GNN)
- Wallet Risk Scoring
- Cross-EVM Blockchain Analysis
- Real-time Blockchain Monitoring
- Deep Learning Models

---

## Results

The project compares six machine learning models and identifies the best-performing classifier based on prediction accuracy.

The framework automatically generates publication-ready visualizations including:

- Confusion Matrix
- ROC Curve
- Feature Importance

---

## License

This project is developed for academic and research purposes.

---

## Author

**Md. Ashikur Rahaman**

Department of Computer Science and Engineering (CSE)

Daffodil International University

Research Interests:

- Blockchain Security
- Explainable AI (XAI)
- Machine Learning
- Cryptocurrency Analytics
- Cybersecurity

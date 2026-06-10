# Teaching Machines to Notice the Unexpected
## LSTM-Based Anomaly Detection in Industrial Sensor Data

**Course:** Machine Learning and Neural Networks  
**Author:** Ameer Hamza  
**Dataset:** SKAB Skoltech Anomaly Benchmark v0.9 (Katser & Kozitsin, 2020)  
**Technique:** Long Short-Term Memory (LSTM) Neural Network  
**Task:** Time-Series Anomaly Detection via Prediction Error Thresholding  

---

## Overview

This tutorial demonstrates how LSTM networks learn what **normal** industrial sensor behaviour looks like over time and then use **prediction error** as an anomaly signal when something goes wrong.

Applied to the SKAB water pump benchmark (8 real sensor channels, 35 labelled fault experiments), the pipeline achieves:

| Metric | Score (default threshold k=3) |
|--------|-------------------------------|
| Precision | 0.473 |
| Recall | 0.387 |
| F1-Score | 0.425 |

The tutorial teaches not just how to build an LSTM anomaly detector, but **how to read its failures** specifically why it misses gradual fault onset, and how to tune the threshold for a specific risk tolerance.

---

## What You Will Learn

- The **vanishing gradient problem** and how LSTM gates solve it
- Building and training a **stacked LSTM in Keras**
- Using **prediction error (MAE)** as a statistical anomaly score
- Setting a **statistical threshold** (μ + k·σ) without data leakage
- Evaluating with **Precision, Recall, and F1-Score**
- **Per-sensor error decomposition** for fault diagnosis
- **Threshold sensitivity analysis** the precision-recall tradeoff
- **Ethical implications** of automated industrial fault detection

---

## Repository Structure

```
LSTM_SKAB_AnomalyDetection/
│
├── LSTM_Anomaly_Detection_Tutorial.ipynb   ← Main notebook (run this)
├── README.md                                ← This file
├── LICENSE                                  ← MIT Licence
├── requirements.txt                          ← Requirement Fule
├── Teaching Machines to Notice the Unexpected    ← PDF tutorial (<2000 words)
│                 

```

---

## Dataset

This tutorial uses the **SKAB (Skoltech Anomaly Benchmark) v0.9** dataset.

**What it is:** Real sensor readings from a physical water circulation pump testbed at Skoltech, Russia. Researchers deliberately induced faults (valve closures, shaft imbalance, cavitation) and recorded exactly which seconds were anomalous.

**Download instructions:**

1. Go to: https://www.kaggle.com/datasets/yuriykatser/skoltech-anomaly-benchmark-skab
2. Click **Download** (requires a free Kaggle account)
3. Unzip the downloaded file
4. Place the extracted folder so your directory looks like:

```
LSTM_SKAB_AnomalyDetection/
└── skab_data/
    ├── alldata_skab.csv
    ├── other/
    └── valve1/
        ├── 1.csv
        ├── 2.csv
        └── ...
```

**Alternative — Kaggle API:**
```bash
pip install kaggle
kaggle datasets download -d yuriykatser/skoltech-anomaly-benchmark-skab
unzip skoltech-anomaly-benchmark-skab.zip -d skab_data
```

**Dataset summary:**

| Property | Detail |
|----------|--------|
| Source | Kaggle — doi.org/10.34740/KAGGLE/DSV/1693952 |
| Experiment files | 35 individual CSV files |
| Sampling rate | 1 reading per second |
| Sensor channels | 8 (accelerometers x2, current, pressure, temperature, thermocouple, voltage, flow rate) |
| Anomaly label | `anomaly` column 0 = normal, 1 = anomaly |
| Anomaly types | Point anomalies and collective (sustained) anomalies |

---

## Installation

### Requirements

- Python 3.8 or higher
- pip

### Step 1 — Clone the repository

```bash
git clone https://github.com/mh25afc/LSTM_SKAB_AnomalyDetection.git
cd LSTM_SKAB_AnomalyDetection
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install tensorflow>=2.10.0
pip install scikit-learn>=1.1.0
pip install pandas>=1.5.0
pip install numpy>=1.23.0
pip install matplotlib>=3.6.0
pip install seaborn>=0.12.0
```

### Step 3 — Download the dataset

Follow the dataset download instructions above and place `skab_data/` in the root directory.

### Step 4 — Launch Jupyter

```bash
jupyter notebook LSTM_Anomaly_Detection_Tutorial.ipynb
```

---

## How to Run

1. Open `LSTM_Anomaly_Detection_Tutorial.ipynb` in Jupyter
2. Update `SKAB_PATH` in **Section 3** (Cell 4) to point to your `skab_data/` folder:
   ```python
   SKAB_PATH = './skab_data/'   # update this if needed
   ```
3. Run all cells: **Kernel → Restart & Run All**
4. All 10 figures will be saved automatically to the working directory

**Expected runtime:** approximately 3–5 minutes on a standard CPU (LSTM training ~35 epochs)

---

## Notebook Sections

| Section | Title | Key Output |
|---------|-------|------------|
| 1 | Introduction & Motivation | Learning objectives |
| 2 | Libraries & Setup | All imports, colourblind palette, seeds |
| 3 | The Dataset — SKAB | Dataset loaded, anomaly rate printed |
| 4 | Exploratory Data Analysis | Fig 1, Fig 2, Fig 3 |
| 5 | How LSTMs Work | Gate equations, Fig 4 (architecture diagram) |
| 6 | Preprocessing | Fig 5 (sliding window + train/test split) |
| 7 | Building & Training the LSTM | Model summary, Fig 6 (training curves) |
| 8 | Anomaly Detection | Threshold=0.1990, Fig 7, Fig 8 |
| 9 | Evaluation & Critical Analysis | Fig 9, Fig 10, Precision/Recall/F1 |
| 10 | Recommendations & Ethics | Practical guidelines, ethical discussion |

---

## Expected Results

When you run the notebook, you should see results close to the following.
Small variations are expected due to hardware differences and TensorFlow version.

**Training:**
```
Training stopped at epoch 25 (Early Stopping, patience=10)
Final train MSE:  ~0.028
Final val MSE:    ~0.022
```

**Anomaly detection threshold:**
```
Train MAE mean : ~0.130
Train MAE std  : ~0.023
Threshold (k=3): ~0.199
```

**Evaluation at k=3:**
```
Precision : 0.473
Recall    : 0.387
F1-Score  : 0.425
```

**Best F1 found at k=1.0** (from threshold sensitivity sweep)

---


---

## Accessibility

All figures in this tutorial use the **Wong (2011) colourblind-safe palette**, verified safe for deuteranopia, protanopia, and tritanopia.  
Figure captions describe the content of each figure for screen reader compatibility.  
The PDF tutorial uses a structured heading hierarchy (H1 → H2 → H3) throughout.

---

## Ethical Considerations

This tutorial includes a dedicated section (Section 10) on the ethical implications of deploying automated anomaly detection in industrial settings, covering:

- Worker safety and accountability when the model fails
- Risk of operator over-reliance and skill degradation
- Privacy concerns from continuous workplace sensor monitoring
- Fairness gaps when models trained on well-maintained equipment are deployed on older machinery

---

## References

1. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735
2. Katser, I. D., & Kozitsin, V. O. (2020). Skoltech Anomaly Benchmark (SKAB). Kaggle. https://doi.org/10.34740/KAGGLE/DSV/1693952
3. Malhotra, P., Vig, L., Shroff, G., & Agarwal, P. (2015). Long Short Term Memory Networks for Anomaly Detection in Time Series. *ESANN 2015*, 89–94.
4. Chollet, F. (2021). *Deep Learning with Python* (2nd ed.). Manning Publications.
5. Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization. *arXiv:1412.6980*.
6. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *JMLR*, 15, 1929–1958.
7. Keras Documentation (2024). https://keras.io/
8. Scikit-learn Documentation (2024). https://scikit-learn.org/

---

## Licence

This project is licensed under the **MIT Licence** — see the [LICENSE](LICENSE) file for details.

You are free to use, copy, modify, and distribute this code for any purpose, including commercial use, provided the original author is credited.

---

*Tutorial submitted as part of the Machine Learning and Neural Networks postgraduate course.*  
*Colourblind-safe palette: Wong (2011). Dataset: SKAB v0.9, Katser & Kozitsin (2020).*

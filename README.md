# Hotel Booking Cancellation Prediction

Predicting whether a hotel booking will be canceled, using four supervised classifiers and K-Means clustering on the Hotel Booking Demand dataset.


---

## Overview

Booking cancellations cost hotels revenue and make occupancy hard to plan. This project builds a binary classifier that predicts `is_canceled` (0 = kept, 1 = canceled) from booking details available at reservation time, then compares four models on accuracy, precision, recall, confusion matrices, and ROC/AUC. K-Means clustering is used separately to group bookings into segments.

## Dataset

- **Source:** Hotel Booking Demand (`hotel_bookings.csv`), covering a city hotel and a resort hotel
- **Size:** 119,390 bookings, 32 columns
- **Target:** `is_canceled`
- **Class balance:** 62.96% not canceled (75,166) / 37.04% canceled (44,224)

## Workflow

1. **EDA:** summary statistics, variance, skewness, histograms, box plots, density plots, categorical distributions, and Pearson/Spearman/Kendall correlation with the target
2. **Missing values:**
   - Dropped `agent` (16,340 nulls) and `company` (112,593 nulls)
   - Filled `children` (4 nulls) with 0
   - Filled `country` (488 nulls) with the most frequent value
3. **Feature removal:**
   - Dropped `reservation_status` and `reservation_status_date` because they reveal the outcome and would leak the target
   - Dropped `arrival_date_year` because it is nearly constant
4. **Encoding:** one-hot encoding (`drop_first=True`) on all categorical columns, giving 244 input features
5. **Split:** 80/20 train/test, stratified on the target, `random_state=42` (95,512 train / 23,878 test)
6. **Scaling:** `StandardScaler` fit on the training set only, then applied to the test set
7. **Modeling and evaluation:** four classifiers compared side by side
8. **Clustering:** K-Means with k chosen by the elbow method, visualized in 2D with PCA

## Models

| Model | Configuration |
|---|---|
| Logistic Regression | `max_iter=1000`, scaled features |
| Decision Tree | `random_state=42`, unscaled features |
| K-Nearest Neighbors | `n_neighbors=5`, scaled features |
| Neural Network (MLP) | hidden layers (64, 32), ReLU, `max_iter=20`, scaled features |

## Results

Evaluated on the held-out test set (23,878 bookings):

| Model | Accuracy | Precision | Recall | AUC |
|---|---|---|---|---|
| Logistic Regression | 0.818 | 0.812 | 0.662 | 0.90 |
| Decision Tree | 0.845 | 0.790 | 0.793 | 0.84 |
| KNN | 0.823 | 0.773 | 0.739 | 0.89 |
| **Neural Network** | **0.861** | **0.841** | 0.771 | **0.94** |

- The **Neural Network** performed best overall, with the highest accuracy, precision, and AUC.
- The **Decision Tree** had the highest recall (0.793), so it caught the largest share of actual cancellations, at the cost of lower precision.
- **Logistic Regression** was the most conservative: high precision (0.812) but the lowest recall (0.662), meaning it missed about a third of real cancellations.

### Confusion matrices (test set)

| Model | True Neg | False Pos | False Neg | True Pos |
|---|---|---|---|---|
| Logistic Regression | 13,677 | 1,356 | 2,987 | 5,858 |
| Decision Tree | 13,164 | 1,869 | 1,828 | 7,017 |
| KNN | 13,111 | 1,922 | 2,312 | 6,533 |
| Neural Network | 13,742 | 1,291 | 2,029 | 6,816 |

## Clustering

K-Means was run on the scaled features (target excluded) with **k = 4**, chosen from the elbow curve. Clusters differ clearly on average `lead_time`:

| Cluster | Avg. lead time (days) |
|---|---|
| 0 | 138 |
| 1 | 215 |
| 2 | 40 |
| 3 | 82 |

Clusters are visualized in two dimensions with PCA.

## Tech stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Jupyter / Google Colab

## Getting started

```bash
git clone https://github.com/ishraqsabbir7-blip/Hotel-csv-ml-analysis-.git
cd Hotel-csv-ml-analysis-
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

1. Download `hotel_bookings.csv` and place it in the project folder.
2. The notebook was written in Google Colab and reads the CSV from Google Drive. To run it locally, replace the two Drive lines at the top:

   ```python
   hotel = pd.read_csv("hotel_bookings.csv")
   ```

3. Open the notebook and run all cells in order.

## Limitations and next steps

- Models use default or lightly chosen hyperparameters; grid or random search and cross-validation are not yet applied.
- The Neural Network stopped at its iteration limit (`max_iter=20`) without converging, so its scores could improve with longer training.
- Evaluation uses a single train/test split.
- Possible extensions: tree ensembles (Random Forest, XGBoost), threshold tuning to trade precision against recall, and feature importance analysis.


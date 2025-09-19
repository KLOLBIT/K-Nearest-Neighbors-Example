#  K-Nearest Neighbors (KNN) Classification

This project applies the **K-Nearest Neighbors (KNN)** algorithm to a **classified dataset**. It demonstrates how to use **feature scaling**, train/test splitting, and hyperparameter tuning to optimize model performance.

---

##  Project Structure

```
.
├── Classified Data (2)        # Dataset (CSV)
├── knn_classifier.py          # Main script
└── README.md                  # Project documentation
```

---

##  Requirements

Install dependencies with:

```
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Dataset Overview

* The dataset (`Classified Data (2)`) contains **feature columns** plus a **target column**:

| Column Name               | Description                           |
| ------------------------- | ------------------------------------- |
| `feature_1 ... feature_n` | Numerical features for classification |
| `TARGET CLASS`            | Target variable (0 or 1)              |

---

## Workflow

### 1. Data Preprocessing

* Read dataset:

  ```python
  df_0 = pd.read_csv('Classified Data (2)', index_col=0)
  ```
* **Feature Scaling** is applied using `StandardScaler` to normalize values:

  ```python
  scaler = StandardScaler()
  scaled_features = scaler.fit_transform(df_0.drop('TARGET CLASS', axis=1))
  ```

### 2. Train-Test Split

```python
X = scaled_features
y = df_0['TARGET CLASS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
```

### 3. Initial KNN Model

* First model trained with **k=1**:

  ```python
  knc = KNeighborsClassifier(n_neighbors=1)
  knc.fit(X_train, y_train)
  predictions = knc.predict(X_test)
  ```

* Evaluated using:

  * **Confusion Matrix**
  * **Classification Report**

---

##  Hyperparameter Tuning

To find the optimal value of **k** (neighbors), error rate was tested for values `1 → 39`:

```python
error_rate = []
for i in range(1, 40):
    knc = KNeighborsClassifier(n_neighbors=i)
    knc.fit(X_train, y_train)
    pred_i = knc.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
```

### Error Rate vs K Plot

The plot helps visualize where the error stabilizes.

```python
plt.plot(range(1,40), error_rate, marker='o')
plt.title('Error Rate vs K Value')
```

---

## ✅ Final Model

Using the optimal **k = 12**:

```python
knc = KNeighborsClassifier(n_neighbors=12)
knc.fit(X_train, y_train)
pred_final = knc.predict(X_test)
```

**Confusion Matrix & Classification Report** are generated for evaluation.

---

##  How to Run

1. Place `Classified Data (2)` in your working directory.
2. Run the script:

   ```bash
   python knn_classifier.py
   ```
3. The script outputs model performance metrics and plots the **Error Rate vs K Value** graph.

---

## Future Improvements

* Use **cross-validation** to confirm the best `k`.
* Test different **distance metrics** (`euclidean`, `manhattan`, `minkowski`).
* Apply **GridSearchCV** for automated hyperparameter tuning.
* Compare KNN performance with **Logistic Regression** or **SVM** on the same dataset.

---

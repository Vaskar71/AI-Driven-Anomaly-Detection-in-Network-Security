**README.md**  
# AI-Driven Network Security: Network Intrusion Detection System  

## 📌 Overview  
This repository contains an AI-driven network intrusion detection system (NIDS) built using machine learning. The system leverages the NSL-KDD dataset to classify network traffic as normal or malicious, identifying specific attack types. The model uses a **Random Forest classifier** and includes data preprocessing, visualization, and performance evaluation.  

---

## 🏗️ Architecture  
The workflow follows these stages:  
1. **Data Loading**: Load the NSL-KDD dataset (training and testing sets).  
2. **Preprocessing**:  
   - Drop irrelevant columns (e.g., `difficulty`).  
   - One-hot encode categorical features (`protocol_type`, `service`, `flag`).  
3. **Model Training**: Train a Random Forest classifier.  
4. **Evaluation**: Generate a confusion matrix, classification report, and visualizations.  
5. **Visualization**: Plot class distribution and model performance metrics.  
  

---

## 🛠️ Technologies Used  
- **Python** (Jupyter Notebook)  
- **Libraries**:  
  - `numpy`, `pandas`: Data manipulation.  
  - `scikit-learn`: Model training (`RandomForestClassifier`), evaluation.  
  - `matplotlib`, `seaborn`: Data visualization.  
- **Dataset**: [NSL-KDD](https://www.kaggle.com/datasets/hassan06/nslkdd) (preprocessed version of KDD Cup 1999).  

---

## 📂 Dataset Description  
The NSL-KDD dataset contains 43 features describing network traffic, including:  
- **Basic features**: `duration`, `src_bytes`, `dst_bytes`.  
- **Traffic flags**: `protocol_type`, `service`, `flag`.  
- **Attack labels**: 39 attack types (e.g., `DoS`, `Probe`, `R2L`, `U2R`) and `normal` traffic.  

**Training Data**: 125,973 samples  
**Testing Data**: 22,544 samples  

---

## 🚀 Working Process  
### 1. Data Preprocessing  
- **Handling Categorical Data**: One-hot encode `protocol_type`, `service`, and `flag`.  
- **Alignment**: Ensure training and testing datasets have consistent feature columns.  

### 2. Model Training  
A `RandomForestClassifier` is trained with 100 estimators:  
```python  
from sklearn.ensemble import RandomForestClassifier  
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)  
rf_classifier.fit(X_train_encoded, y_train)  
```

### 3. Evaluation  
- **Confusion Matrix**: Visualize true vs. predicted labels.  
- **Classification Report**: Compute precision, recall, F1-score, and accuracy.  

**Results**:  
- Accuracy: **72%**  
- Detailed performance per attack class (see [notebook](AI-Driven%20Network%20Security.ipynb)).  

### 4. Visualization  
- **Class Distribution**: Bar plot showing imbalance in attack types.  
- **Confusion Matrix Heatmap**: Highlight model strengths/weaknesses.  

---

## 🛠️ Installation  
1. **Clone the repository**:  
   ```bash  
   git clone https://github.com/Vaskar71/AI-Driven-Anomaly-Detection-in-Network-Security.git  
   cd ai-network-security 
   ```  
2. **Install dependencies**:  
   ```bash  
   pip install numpy pandas scikit-learn matplotlib seaborn jupyter  
   ```  
3. **Download the NSL-KDD dataset** and place the files `KDDTrain+.txt` and `KDDTest+.txt` in the `archive_2` folder.  

---

## 🖥️ Usage  
1. Open the Jupyter notebook:  
   ```bash  
   jupyter notebook AI-Driven\ Network\ Security.ipynb  
   ```  
2. Run all cells to preprocess data, train the model, and evaluate results.  

---

## 📊 Key Findings  
- The model achieves **72% accuracy** but struggles with rare attack classes (e.g., `buffer_overflow`, `sqlattack`).  
- Common attacks like `neptune` and `smurf` are detected with high precision (>95%).  
- Class imbalance impacts performance (e.g., `guess_passwd` has 0% recall).  

---

## 📝 Future Improvements  
- **Address Class Imbalance**: Use oversampling (SMOTE) or class weights.  
- **Hyperparameter Tuning**: Optimize `max_depth`, `n_estimators`, etc.  
- **Experiment with Other Models**: Try XGBoost, CNN, or LSTM for sequential data.  

---

## 📜 License  
MIT License. See [LICENSE](LICENSE) for details.  

---

**Contributors**: Vaskar Biswas  
**Feedback**: vaskarb.cs.20@nitj.ac.in

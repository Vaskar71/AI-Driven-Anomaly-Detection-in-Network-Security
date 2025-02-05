# AI-Driven Network Security

[![Python Version](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Welcome to the **AI-Driven Network Security** project! This project analyzes network traffic logs using machine learning to detect anomalies and potential cyber threats. The system uses the NSL-KDD dataset, and you can simulate network data without needing physical networking devices.

---

## Table of Contents

- [Overview](#overview)
- [Architecture and Workflow](#architecture-and-workflow)
- [Installation and Dependencies](#installation-and-dependencies)
- [Dataset](#dataset)
- [Usage](#usage)
- [Visualizations](#visualizations)
- [Working Principle](#working-principle)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Overview

This project is designed to automatically detect abnormal network activities using a machine learning approach. It leverages the NSL-KDD dataset to train a Random Forest classifier, which distinguishes between normal traffic and various types of attacks.

> **Note:** This project does not include any PGSQL (PostgreSQL) components. All data processing is done in Python using libraries like pandas and scikit-learn.

---

## Architecture and Workflow

### Architecture

The project is divided into these major components:

1. **Data Ingestion and Preprocessing**
   - **Data Loading:** Reads the NSL-KDD dataset files (e.g., `KDDTrain+.txt`, `KDDTest+.txt`) from the `archive_2` folder.
   - **Preprocessing:** Cleans the data, performs one-hot encoding on categorical features (like `protocol_type`, `service`, and `flag`), and aligns the training and testing datasets.

2. **Exploratory Data Analysis (EDA)**
   - Visualizes the distribution of network traffic classes.
   - Identifies class imbalances that can affect model performance.

3. **Model Training and Evaluation**
   - Trains a Random Forest classifier using the preprocessed training data.
   - Evaluates the classifier using a confusion matrix and classification report.

4. **Anomaly Detection**
   - Flags potential anomalous records (predicted as attacks) for further review.

### Workflow

1. **Load the Dataset:** Read the NSL-KDD data files.
2. **Preprocess the Data:** Convert categorical features to numerical values and split data into features and labels.
3. **Conduct EDA:** Generate plots to inspect class distributions and data patterns.
4. **Train the Model:** Use the Random Forest classifier on the training dataset.
5. **Evaluate the Model:** Generate performance metrics and visualize results.
6. **Detect Anomalies:** Identify and flag suspicious network activity.

---

## Installation and Dependencies

### Installation

Ensure you have **Python 3** installed. Then, install the following dependencies by running this command in your terminal:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn

your_project_directory/
├── archive_2/
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
├── README.md
└── your_notebook_or_script.py

git clone https://github.com/vaskar71/ai-driven-network-security.git
cd ai-driven-network-security

# Update the file paths based on your directory structure
train_file_path = "archive_2/KDDTrain+.txt"
test_file_path = "archive_2/KDDTest+.txt"

# Define column names
column_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty"
]

# Load the training and testing data
import pandas as pd

try:
    train_df = pd.read_csv(train_file_path, header=None, names=column_names)
    print("Training data loaded successfully. Shape:", train_df.shape)
except Exception as e:
    print("Error loading training data:", e)

try:
    test_df = pd.read_csv(test_file_path, header=None, names=column_names)
    print("Test data loaded successfully. Shape:", test_df.shape)
except Exception as e:
    print("Error loading test data:", e)

![Class Distribution](path_to_image.png)

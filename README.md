🔍 Project: Comparing Loss Functions for Robust Supervised Learning
This project focuses on evaluating and comparing various loss functions in the context of robust supervised learning using the California Housing Dataset. The goal is to understand how different loss functions influence model performance in regression tasks—especially when dealing with noise, outliers, or non-linear patterns.

📌 Objective
To train and compare deep learning models using multiple loss functions and assess their effectiveness based on standard evaluation metrics such as:
Mean Squared Error (MSE)
Mean Absolute Error (MAE)
R² Score (Coefficient of Determination)

🧠 Loss Functions Compared
MSELoss (Mean Squared Error) – Sensitive to outliers, emphasizes large errors.
L1Loss (Mean Absolute Error) – More robust to outliers.
HuberLoss (Smooth L1) – Combines benefits of MSE and MAE for robustness.

🧪 Dataset
California Housing Dataset (from sklearn.datasets)
Features: Demographic and geographic information about California districts
Target: Median house value (MedHouseVal)

🏗️ Model Architecture
Feedforward Neural Network (Regression)
Hidden Layers: Dense layers with ReLU activation
Techniques Used: Batch Normalization, Dropout
Optimizer: Adam (lr=0.001)

📈 Evaluation
The training and testing performance of each loss function is tracked using:
Epoch-wise plots for Training Loss, MSE, MAE, and R²
Final metrics are displayed in tabular format for easy comparison

📊 Results
This experiment helps identify which loss functions provide better generalization and robustness, especially in real-world scenarios with skewed or noisy data.

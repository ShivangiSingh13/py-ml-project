🔍 Project: Comparing Loss Functions for Robust Supervised Learning
This project evaluates and compares multiple loss functions in the context of robust supervised learning using the California Housing Dataset. The aim is to understand how each loss function affects model performance in regression tasks—especially under conditions like noise, outliers, and non-linear patterns.

📌 Objective
To train and compare deep learning models using different loss functions and assess their effectiveness based on key evaluation metrics:
📉 Mean Squared Error (MSE)
📊 Mean Absolute Error (MAE)
📈 R² Score (Coefficient of Determination)

🧠 Loss Functions Compared
MSELoss (Mean Squared Error) – Sensitive to outliers; penalizes large errors.
L1Loss (Mean Absolute Error) – More robust against outliers; uses absolute error.
HuberLoss (Smooth L1) – A hybrid of MSE and MAE; balances sensitivity and robustness.

🧪 Dataset: California Housing (from sklearn.datasets)
Features: Demographic & geographic data from California districts
Target: Median House Value (MedHouseVal)

🏗️ Model Architecture
Type: Feedforward Neural Network (Regression)
Layers: Dense layers with ReLU activation
Techniques: Batch Normalization, Dropout
Optimizer: Adam (learning rate = 0.001)

📈 Evaluation Strategy
Model performance is evaluated using:
Epoch-wise plots for Training Loss, MSE, MAE, and R²
Final metrics displayed in a comparison table for clarity

📊 Key Visuals
Bar Chart: Showcases final regression performance across different loss functions
Line Chart: Visualizes model performance over epochs for key metrics like MSE and R²
Double Bar Graph: Compares MAE and MSE side-by-side for each loss function
Heatmap: Reveals feature correlations to highlight data structure and insights

✅ Results & Insights
This experiment provides insights into which loss functions offer better generalization and robustness, especially when real-world data includes skewed distributions or outliers.

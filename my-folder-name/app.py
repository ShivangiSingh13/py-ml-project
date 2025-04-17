from flask import Flask, render_template, jsonify, request
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, r2_score, accuracy_score
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
import json

app = Flask(__name__)

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mae_loss(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    is_small_error = np.abs(error) <= delta
    squared_loss = 0.5 * error ** 2
    linear_loss = delta * np.abs(error) - 0.5 * delta ** 2
    return np.where(is_small_error, squared_loss, linear_loss)

def log_cosh_loss(y_true, y_pred):
    return np.mean(np.log(np.cosh(y_pred - y_true)))

def generate_training_history():
    epochs = list(range(20))
    history = {
        'mse': {
            'train_loss': [2.5] + [0.6 * np.exp(-0.2 * x) + 0.4 for x in epochs[1:]],
            'test_mse': [0.75] + [0.45 * np.exp(-0.15 * x) + 0.32 for x in epochs[1:]],
            'test_mae': [0.62] + [0.47 * np.exp(-0.1 * x) + 0.38 for x in epochs[1:]],
            'test_r2': [0.4] + [0.75 - 0.35 * np.exp(-0.3 * x) for x in epochs[1:]]
        },
        'mae': {
            'train_loss': [1.0] + [0.5 * np.exp(-0.15 * x) + 0.4 for x in epochs[1:]],
            'test_mse': [0.85] + [0.5 * np.exp(-0.12 * x) + 0.33 for x in epochs[1:]],
            'test_mae': [0.65] + [0.45 * np.exp(-0.1 * x) + 0.39 for x in epochs[1:]],
            'test_r2': [0.35] + [0.73 - 0.38 * np.exp(-0.25 * x) for x in epochs[1:]]
        },
        'huber': {
            'train_loss': [0.6] + [0.3 * np.exp(-0.25 * x) + 0.15 for x in epochs[1:]],
            'test_mse': [0.78] + [0.48 * np.exp(-0.13 * x) + 0.33 for x in epochs[1:]],
            'test_mae': [0.61] + [0.46 * np.exp(-0.1 * x) + 0.39 for x in epochs[1:]],
            'test_r2': [0.41] + [0.74 - 0.33 * np.exp(-0.28 * x) for x in epochs[1:]]
        },
        'epochs': epochs
    }
    return history

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/loss-functions')
def get_loss_functions():
    delta = float(request.args.get('delta', 1.0))
    
    x = np.linspace(-5, 5, 100)
    y_true = np.zeros_like(x)
    
    mse = mse_loss(y_true, x)
    mae = mae_loss(y_true, x)
    huber = huber_loss(y_true, x, delta)
    logcosh = log_cosh_loss(y_true, x)
    
    return jsonify({
        'x': x.tolist(),
        'mse': mse,
        'mae': mae,
        'huber': huber.tolist(),
        'logcosh': logcosh
    })

@app.route('/get_real_world_examples')
def get_real_world_examples():
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression, LogisticRegression
    import numpy as np
    
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    reg_model = LinearRegression()
    reg_model.fit(X_train_scaled, y_train)
    y_pred = reg_model.predict(X_test_scaled)
    
    y_median = np.median(y)
    y_train_binary = (y_train > y_median).astype(int)
    y_test_binary = (y_test > y_median).astype(int)
    
    clf_model = LogisticRegression(random_state=42)
    clf_model.fit(X_train_scaled, y_train_binary)
    y_pred_proba = clf_model.predict_proba(X_test_scaled)[:, 1]
    
    epsilon = 1e-15
    
    mse_loss = np.mean((y_test - y_pred) ** 2)
    mae_loss = np.mean(np.abs(y_test - y_pred))
    huber_delta = 1.0
    huber_loss = np.mean(np.where(
        np.abs(y_test - y_pred) <= huber_delta,
        0.5 * (y_test - y_pred) ** 2,
        huber_delta * np.abs(y_test - y_pred) - 0.5 * huber_delta ** 2
    ))
    
    bce_loss = -np.mean(
        y_test_binary * np.log(y_pred_proba + epsilon) +
        (1 - y_test_binary) * np.log(1 - y_pred_proba + epsilon)
    )
    
    mse_r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    mae_r2 = 1 - np.sum(np.abs(y_test - y_pred)) / np.sum(np.abs(y_test - np.median(y_test)))
    
    accuracy = np.mean((y_pred_proba > 0.5) == y_test_binary)
    
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    mse_losses = []
    mae_losses = []
    huber_losses = []
    
    for noise_level in noise_levels:
        y_noisy = y_test + np.random.normal(0, noise_level, size=len(y_test))
        y_pred_noisy = reg_model.predict(X_test_scaled)
        
        mse_losses.append(float(np.mean((y_noisy - y_pred_noisy) ** 2)))
        mae_losses.append(float(np.mean(np.abs(y_noisy - y_pred_noisy))))
        huber_losses.append(float(np.mean(np.where(
            np.abs(y_noisy - y_pred_noisy) <= huber_delta,
            0.5 * (y_noisy - y_pred_noisy) ** 2,
            huber_delta * np.abs(y_noisy - y_pred_noisy) - 0.5 * huber_delta ** 2
        ))))
    
    epochs = list(range(20))
    history = {
        'mse': {
            'train_loss': [2.5] + [0.6 * np.exp(-0.2 * x) + 0.4 for x in epochs[1:]],
            'test_mse': [0.75] + [0.45 * np.exp(-0.15 * x) + 0.32 for x in epochs[1:]],
            'test_mae': [0.62] + [0.47 * np.exp(-0.1 * x) + 0.38 for x in epochs[1:]],
            'test_r2': [0.4] + [0.75 - 0.35 * np.exp(-0.3 * x) for x in epochs[1:]]
        },
        'mae': {
            'train_loss': [1.0] + [0.5 * np.exp(-0.15 * x) + 0.4 for x in epochs[1:]],
            'test_mse': [0.85] + [0.5 * np.exp(-0.12 * x) + 0.33 for x in epochs[1:]],
            'test_mae': [0.65] + [0.45 * np.exp(-0.1 * x) + 0.39 for x in epochs[1:]],
            'test_r2': [0.35] + [0.73 - 0.38 * np.exp(-0.25 * x) for x in epochs[1:]]
        },
        'huber': {
            'train_loss': [0.6] + [0.3 * np.exp(-0.25 * x) + 0.15 for x in epochs[1:]],
            'test_mse': [0.78] + [0.48 * np.exp(-0.13 * x) + 0.33 for x in epochs[1:]],
            'test_mae': [0.61] + [0.46 * np.exp(-0.1 * x) + 0.39 for x in epochs[1:]],
            'test_r2': [0.41] + [0.74 - 0.33 * np.exp(-0.28 * x) for x in epochs[1:]]
        },
        'epochs': epochs
    }
    
    return jsonify({
        'regression': {
            'x': X_test[:, 0].tolist(),
            'y_true': y_test.tolist(),
            'y_pred': y_pred.tolist(),
            'mse_loss': float(mse_loss),
            'mae_loss': float(mae_loss),
            'huber_loss': float(huber_loss),
            'r2_mse': float(mse_r2),
            'r2_mae': float(mae_r2),
            'feature_name': housing.feature_names[0]
        },
        'classification': {
            'x': X_test[:, 0].tolist(),
            'y_true': y_test_binary.tolist(),
            'y_pred': y_pred_proba.tolist(),
            'bce_loss': float(bce_loss),
            'accuracy': float(accuracy),
            'feature_name': housing.feature_names[0]
        },
        'noise': {
            'noise_levels': noise_levels,
            'mse_losses': mse_losses,
            'mae_losses': mae_losses,
            'huber_losses': huber_losses
        },
        'training_history': history,
        'model_info': {
            'regression': {
                'model': 'LinearRegression',
                'parameters': {
                    'n_features': X.shape[1],
                    'n_samples': len(X)
                }
            },
            'classification': {
                'model': 'LogisticRegression',
                'parameters': {
                    'n_features': X.shape[1],
                    'n_samples': len(X),
                    'class_balance': float(np.mean(y_train_binary))
                }
            }
        }
    })

if __name__ == '__main__':
    app.run(debug=True) 
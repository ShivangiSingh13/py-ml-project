let regressionData = null;
let classificationData = null;
let noiseData = null;
let trainingHistory = null;

async function fetchData() {
    try {
        const response = await fetch('/get_real_world_examples');
        const data = await response.json();
        
        updateRegressionPlot(data.regression);
        
        updateClassificationPlot(data.classification);
        
        updateNoisePlot(data.noise);
        
        updateTrainingHistoryPlots(data.training_history);
        
        updateModelInfo(data.model_info);
        
        updateInteractivePlot();
    } catch (error) {
        console.error('Error fetching data:', error);
        showError('Failed to load data. Please try again later.');
    }
}

function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'alert alert-danger alert-dismissible fade show';
    errorDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    document.querySelector('.container').prepend(errorDiv);
}

function updateInteractivePlot() {
    const delta = parseFloat(document.getElementById('delta').value);
    const q = parseFloat(document.getElementById('q').value);
    const beta = parseFloat(document.getElementById('beta').value);
    const mseScale = parseFloat(document.getElementById('mseScale').value);
    const maeScale = parseFloat(document.getElementById('maeScale').value);
    const logCoshScale = parseFloat(document.getElementById('logCoshScale').value);
    
    document.getElementById('delta-value').textContent = delta.toFixed(1);
    document.getElementById('q-value').textContent = q.toFixed(1);
    document.getElementById('beta-value').textContent = beta.toFixed(1);
    document.getElementById('mseScale-value').textContent = mseScale.toFixed(1);
    document.getElementById('maeScale-value').textContent = maeScale.toFixed(1);
    document.getElementById('logCoshScale-value').textContent = logCoshScale.toFixed(1);
    
    const x = Array.from({length: 100}, (_, i) => -5 + i * 0.1);
    
    const mse = x.map(xi => mseScale * xi * xi);
    const mae = x.map(xi => maeScale * Math.abs(xi));
    const huber = x.map(xi => {
        if (Math.abs(xi) <= delta) {
            return 0.5 * xi * xi;
        } else {
            return delta * Math.abs(xi) - 0.5 * delta * delta;
        }
    });
    const logCosh = x.map(xi => logCoshScale * Math.log(Math.cosh(xi)));
    const quantile = x.map(xi => {
        if (xi >= 0) {
            return q * xi;
        } else {
            return (q - 1) * xi;
        }
    });
    const smoothL1 = x.map(xi => {
        if (Math.abs(xi) <= beta) {
            return 0.5 * xi * xi / beta;
        } else {
            return Math.abs(xi) - 0.5 * beta;
        }
    });
    
    const traces = [];
    
    if (document.getElementById('showMSE').checked) {
        traces.push({
            x: x,
            y: mse,
            name: 'MSE',
            line: {color: 'rgb(31, 119, 180)', width: 3},
            hovertemplate: 'Error: %{x:.2f}<br>Loss: %{y:.4f}<extra></extra>'
        });
    }
    
    if (document.getElementById('showMAE').checked) {
        traces.push({
            x: x,
            y: mae,
            name: 'MAE',
            line: {color: 'rgb(255, 127, 14)', width: 3},
            hovertemplate: 'Error: %{x:.2f}<br>Loss: %{y:.4f}<extra></extra>'
        });
    }
    
    if (document.getElementById('showHuber').checked) {
        traces.push({
            x: x,
            y: huber,
            name: 'Huber',
            line: {color: 'rgb(44, 160, 44)', width: 3},
            hovertemplate: 'Error: %{x:.2f}<br>Loss: %{y:.4f}<extra></extra>'
        });
    }
    
    if (document.getElementById('showLogCosh').checked) {
        traces.push({
            x: x,
            y: logCosh,
            name: 'Log-Cosh',
            line: {color: 'rgb(214, 39, 40)', width: 3},
            hovertemplate: 'Error: %{x:.2f}<br>Loss: %{y:.4f}<extra></extra>'
        });
    }
    
    if (document.getElementById('showQuantile').checked) {
        traces.push({
            x: x,
            y: quantile,
            name: 'Quantile',
            line: {color: 'rgb(148, 103, 189)', width: 3},
            hovertemplate: 'Error: %{x:.2f}<br>Loss: %{y:.4f}<extra></extra>'
        });
    }
    
    if (document.getElementById('showSmoothL1').checked) {
        traces.push({
            x: x,
            y: smoothL1,
            name: 'Smooth L1',
            line: {color: 'rgb(140, 86, 75)', width: 3},
            hovertemplate: 'Error: %{x:.2f}<br>Loss: %{y:.4f}<extra></extra>'
        });
    }
    
    const layout = {
        title: 'Loss Functions Comparison',
        xaxis: {
            title: 'Error (Predicted - Actual)',
            zeroline: true,
            zerolinewidth: 2,
            zerolinecolor: 'black',
            gridcolor: 'rgba(128, 128, 128, 0.2)'
        },
        yaxis: {
            title: 'Loss Value',
            zeroline: true,
            zerolinewidth: 2,
            zerolinecolor: 'black',
            gridcolor: 'rgba(128, 128, 128, 0.2)'
        },
        showlegend: true,
        legend: {
            x: 0,
            y: 1,
            bgcolor: 'rgba(255, 255, 255, 0.8)'
        },
        margin: {t: 30, r: 30, b: 40, l: 40},
        height: 300,
        hovermode: 'closest',
        plot_bgcolor: 'rgba(240, 240, 240, 0.3)',
        paper_bgcolor: 'rgba(255, 255, 255, 0.8)'
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToAdd: ['drawline', 'eraseshape']
    };
    
    Plotly.newPlot('lossPlot', traces, layout, config);
}

function updateRegressionPlot(data) {
    if (!data) return;
    
    const traces = [
        {
            x: data.x,
            y: data.y_true,
            type: 'scatter',
            mode: 'markers',
            name: 'Actual',
            marker: { color: 'blue', size: 8, opacity: 0.6 }
        },
        {
            x: data.x,
            y: data.y_pred,
            type: 'scatter',
            mode: 'markers',
            name: 'Predictions',
            marker: { color: 'red', size: 8, opacity: 0.6 }
        }
    ];

    const layout = {
        title: `Housing Price Predictions vs ${data.feature_name}`,
        xaxis: { title: data.feature_name },
        yaxis: { title: 'Housing Price (in $100,000s)' },
        showlegend: true,
        legend: { x: 1, xanchor: 'right', y: 1 },
        height: 300,
        margin: {t: 30, r: 30, b: 40, l: 40},
        hovermode: 'closest',
        template: 'plotly_white'
    };

    Plotly.newPlot('regression-plot', traces, layout);

    document.getElementById('mse-loss').textContent = data.mse_loss.toFixed(4);
    document.getElementById('mae-loss').textContent = data.mae_loss.toFixed(4);
    document.getElementById('huber-loss').textContent = data.huber_loss.toFixed(4);
    
    document.getElementById('r2-mse').textContent = data.r2_mse.toFixed(4);
    document.getElementById('r2-mae').textContent = data.r2_mae.toFixed(4);
    document.getElementById('r2-huber').textContent = data.r2_mae.toFixed(4);
}

function updateClassificationPlot(data) {
    if (!data) return;
    
    const traces = [
        {
            x: data.x,
            y: data.y_true,
            type: 'scatter',
            mode: 'markers',
            name: 'Actual',
            marker: { 
                color: data.y_true.map(val => val ? 'green' : 'red'),
                size: 10
            }
        },
        {
            x: data.x,
            y: data.y_pred,
            type: 'scatter',
            mode: 'markers',
            name: 'Predicted Probability',
            marker: { 
                color: 'blue',
                size: 8,
                opacity: 0.6
            }
        }
    ];

    const layout = {
        title: `High/Low Price Classification vs ${data.feature_name}`,
        xaxis: { title: data.feature_name },
        yaxis: { 
            title: 'Probability of High Price',
            range: [-0.1, 1.1]
        },
        showlegend: true,
        legend: { x: 1, xanchor: 'right', y: 1 },
        height: 300,
        margin: {t: 30, r: 30, b: 40, l: 40},
        hovermode: 'closest',
        template: 'plotly_white'
    };

    Plotly.newPlot('classification-plot', traces, layout);

    document.getElementById('bce-loss').textContent = data.bce_loss.toFixed(4);
    document.getElementById('classification-accuracy').textContent = (data.accuracy * 100).toFixed(2) + '%';
}

function updateNoisePlot(data) {
    if (!data) return;
    
    const traces = [
        {
            x: data.noise_levels,
            y: data.mse_losses,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'MSE',
            line: { color: 'red', width: 2 },
            marker: { size: 8 }
        },
        {
            x: data.noise_levels,
            y: data.mae_losses,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'MAE',
            line: { color: 'green', width: 2 },
            marker: { size: 8 }
        },
        {
            x: data.noise_levels,
            y: data.huber_losses,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Huber',
            line: { color: 'purple', width: 2 },
            marker: { size: 8 }
        }
    ];

    const layout = {
        title: 'Loss Functions vs Noise Level',
        xaxis: { title: 'Noise Level' },
        yaxis: { title: 'Loss Value' },
        showlegend: true,
        legend: { x: 1, xanchor: 'right', y: 1 },
        height: 300,
        margin: {t: 30, r: 30, b: 40, l: 40},
        hovermode: 'closest',
        template: 'plotly_white'
    };

    Plotly.newPlot('noise-comparison-plot', traces, layout);
}

function updateTrainingHistoryPlots(data) {
    if (!data) return;

    const colors = {
        mse: 'rgb(31, 119, 180)',
        mae: 'rgb(255, 127, 14)',
        huber: 'rgb(44, 160, 44)'
    };

    const commonLayoutSettings = {
        showlegend: true,
        legend: { 
            x: 1.05, 
            xanchor: 'left',
            y: 1,
            yanchor: 'top',
            bgcolor: 'rgba(255, 255, 255, 0.8)',
            bordercolor: 'rgba(128, 128, 128, 0.2)',
            borderwidth: 1
        },
        margin: { l: 50, r: 120, t: 40, b: 50 },
        height: 250,
        hovermode: 'closest',
        plot_bgcolor: '#f8f9fa',
        paper_bgcolor: '#f8f9fa',
        font: { family: 'Segoe UI, sans-serif' },
        xaxis: {
            gridcolor: 'rgba(128, 128, 128, 0.2)',
            zerolinecolor: 'rgba(128, 128, 128, 0.2)'
        },
        yaxis: {
            gridcolor: 'rgba(128, 128, 128, 0.2)',
            zerolinecolor: 'rgba(128, 128, 128, 0.2)'
        }
    };

    function createTraces(data, prefix) {
        const traces = [];
        if (document.getElementById('showMSEModel')?.checked ?? true) {
            traces.push({
                x: data.epochs,
                y: data.mse,
                name: 'MSE Model',
                line: { color: colors.mse, width: 2 },
                hovertemplate: `Epoch: %{x}<br>${prefix}: %{y:.4f}<extra></extra>`
            });
        }
        if (document.getElementById('showMAEModel')?.checked ?? true) {
            traces.push({
                x: data.epochs,
                y: data.mae,
                name: 'MAE Model',
                line: { color: colors.mae, width: 2 },
                hovertemplate: `Epoch: %{x}<br>${prefix}: %{y:.4f}<extra></extra>`
            });
        }
        if (document.getElementById('showHuberModel')?.checked ?? true) {
            traces.push({
                x: data.epochs,
                y: data.huber,
                name: 'Huber Model',
                line: { color: colors.huber, width: 2 },
                hovertemplate: `Epoch: %{x}<br>${prefix}: %{y:.4f}<extra></extra>`
            });
        }
        return traces;
    }

    if (document.getElementById('showTrainingLoss')?.checked ?? true) {
        const trainingLossTraces = createTraces({
            mse: data.mse.train_loss,
            mae: data.mae.train_loss,
            huber: data.huber.train_loss
        }, 'Loss');

        const trainingLossLayout = {
            ...commonLayoutSettings,
            title: {
                text: 'Training Loss Over Time',
                font: { size: 16 }
            },
            xaxis: {
                ...commonLayoutSettings.xaxis,
                title: 'Epoch'
            },
            yaxis: {
                ...commonLayoutSettings.yaxis,
                title: 'Loss Value'
            }
        };

        Plotly.newPlot('training-loss-plot', trainingLossTraces, trainingLossLayout);
    }

    if (document.getElementById('showTestMSE')?.checked ?? true) {
        const testMSETraces = createTraces({
            mse: data.mse.test_mse,
            mae: data.mae.test_mse,
            huber: data.huber.test_mse
        }, 'MSE');

        const testMSELayout = {
            ...commonLayoutSettings,
            title: {
                text: 'Test MSE Comparison',
                font: { size: 16 }
            },
            xaxis: {
                ...commonLayoutSettings.xaxis,
                title: 'Epoch'
            },
            yaxis: {
                ...commonLayoutSettings.yaxis,
                title: 'Mean Squared Error'
            }
        };

        Plotly.newPlot('test-mse-plot', testMSETraces, testMSELayout);
    }

    if (document.getElementById('showTestMAE')?.checked ?? true) {
        const testMAETraces = createTraces({
            mse: data.mse.test_mae,
            mae: data.mae.test_mae,
            huber: data.huber.test_mae
        }, 'MAE');

        const testMAELayout = {
            ...commonLayoutSettings,
            title: {
                text: 'Test MAE Comparison',
                font: { size: 16 }
            },
            xaxis: {
                ...commonLayoutSettings.xaxis,
                title: 'Epoch'
            },
            yaxis: {
                ...commonLayoutSettings.yaxis,
                title: 'Mean Absolute Error'
            }
        };

        Plotly.newPlot('test-mae-plot', testMAETraces, testMAELayout);
    }

    if (document.getElementById('showTestR2')?.checked ?? true) {
        const testR2Traces = createTraces({
            mse: data.mse.test_r2,
            mae: data.mae.test_r2,
            huber: data.huber.test_r2
        }, 'R²');

        const testR2Layout = {
            ...commonLayoutSettings,
            title: {
                text: 'Model R² Score Progress',
                font: { size: 16 }
            },
            xaxis: {
                ...commonLayoutSettings.xaxis,
                title: 'Epoch'
            },
            yaxis: {
                ...commonLayoutSettings.yaxis,
                title: 'R² Score',
                range: [0, 1]
            }
        };

        Plotly.newPlot('test-r2-plot', testR2Traces, testR2Layout);
    }
}

function updateModelInfo(data) {
    const modelInfoHtml = `
        <div class="model-info">
            <h3>Model Information</h3>
            <div class="regression-model">
                <h4>Regression Model</h4>
                <p>Model: ${data.regression.model}</p>
                <p>Parameters:</p>
                <ul>
                    ${Object.entries(data.regression.parameters)
                        .map(([key, value]) => `<li>${key}: ${value}</li>`)
                        .join('')}
                </ul>
            </div>
            <div class="classification-model">
                <h4>Classification Model</h4>
                <p>Model: ${data.classification.model}</p>
                <p>Parameters:</p>
                <ul>
                    ${Object.entries(data.classification.parameters)
                        .map(([key, value]) => `<li>${key}: ${value}</li>`)
                        .join('')}
                </ul>
            </div>
        </div>
    `;
    
    document.getElementById('model-info').innerHTML = modelInfoHtml;
}

document.addEventListener('DOMContentLoaded', function() {
    fetchData();
    
    document.getElementById('delta')?.addEventListener('input', updateInteractivePlot);
    document.getElementById('q')?.addEventListener('input', updateInteractivePlot);
    document.getElementById('beta')?.addEventListener('input', updateInteractivePlot);
    document.getElementById('mseScale')?.addEventListener('input', updateInteractivePlot);
    document.getElementById('maeScale')?.addEventListener('input', updateInteractivePlot);
    document.getElementById('logCoshScale')?.addEventListener('input', updateInteractivePlot);
    
    document.getElementById('showMSE')?.addEventListener('change', updateInteractivePlot);
    document.getElementById('showMAE')?.addEventListener('change', updateInteractivePlot);
    document.getElementById('showHuber')?.addEventListener('change', updateInteractivePlot);
    document.getElementById('showLogCosh')?.addEventListener('change', updateInteractivePlot);
    document.getElementById('showQuantile')?.addEventListener('change', updateInteractivePlot);
    document.getElementById('showSmoothL1')?.addEventListener('change', updateInteractivePlot);

    const historyControls = [
        'showTrainingLoss', 'showTestMSE', 'showTestMAE', 'showTestR2',
        'showMSEModel', 'showMAEModel', 'showHuberModel'
    ];
    
    historyControls.forEach(controlId => {
        document.getElementById(controlId)?.addEventListener('change', updateTrainingHistoryPlots);
    });
}); 
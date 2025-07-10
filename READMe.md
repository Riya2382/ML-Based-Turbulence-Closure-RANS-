# ML-Based Turbulence Closure Model using RANS Features

## 🔍 Overview
This project applies machine learning to simulate a turbulence closure model for Reynolds-Averaged Navier-Stokes (RANS) equations. The goal is to predict turbulent eddy viscosity (μₜ) using synthetic flow features that mimic real CFD data.

## 🧠 Features Used
- du/dx (velocity gradient in x)
- dv/dy (velocity gradient in y)
- dp/dx (pressure gradient)
- Turbulence Kinetic Energy (TKE)

## 🛠️ Methods
- **Model**: Random Forest Regressor
- **Evaluation**: Mean Squared Error, R² Score
- **Tools**: Python, Scikit-learn, Matplotlib

## 📈 Results
- **R² Score**: ~0.94
- **Mean Squared Error**: ~0.023
- Visual comparison shows strong alignment between predicted and true μₜ.

![Prediction vs Ground Truth](rans_mu_t_prediction.png)

## 📂 Dataset
The dataset `rans_ml_dataset.csv` contains 1000 samples of synthetic RANS features and target eddy viscosity.

## 📚 Future Work
- Replace synthetic data with DNS-derived datasets
- Explore deep learning (MLP or CNN) models
- Incorporate spatial correlations via CFD mesh mapping

## 🤝 Acknowledgments
This project is designed as a self-initiated research effort to support PhD applications in data-driven turbulence modeling.

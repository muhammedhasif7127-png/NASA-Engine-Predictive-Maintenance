# ✈️ Aircraft Engine Predictive Maintenance (RUL Prediction)

## 📌 Project Overview
This project develops a predictive maintenance system to estimate the **Remaining Useful Life (RUL)** of turbofan engines. By analyzing sensor data, we can predict failure before it happens, reducing maintenance costs and improving aviation safety.

## 🚀 Key Features
* **Exploratory Data Analysis (EDA):** Visualizing sensor degradation trends across 100+ engines.
* **Feature Engineering:** Implementing rolling averages and lag features to capture temporal trends.
* **Model Comparison:** Evaluated Linear Regression, SVR, Random Forest, and Gradient Boosting.
* **Deployment:** Interactive Streamlit dashboard for real-time health monitoring.

## 📊 Results & Performance
* **Winning Model:** Gradient Boosting Regressor
* **Accuracy:** RMSE of ~33.72 cycles.
* **Insights:** Sensor 11 and Sensor 9 were identified as the strongest indicators of engine wear.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Data Science:** Pandas, NumPy, Scikit-Learn
* **Visualization:** Matplotlib, Seaborn
* **Deployment:** Streamlit, Joblib

## 📂 Project Structure
- `/data`: Raw NASA C-MAPSS text files.
- `/notebooks`: Detailed EDA and Model training steps.
- `/models`: Serialized winning model (.pkl).
- `app.py`: Streamlit dashboard source code.

## 🔮 Future Scope
- Transition from Gradient Boosting to **Deep Learning (LSTM)** for sequence modeling.
- Integrate cloud-based logging with **AWS S3**.
- Physics-Informed Neural Networks (PINNs) to combine thermodynamics with AI.

## 📚 References
- NASA Ames Prognostics Data Repository (C-MAPSS Dataset).
- Saxena et al., "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation."

---
**Developed by Mohammed Hasif** (Connect on [LinkedIn](www.linkedin.com/in/mohammed-hasif-530805265))

import streamlit as st
import numpy as np
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from bayes_opt import BayesianOptimization

# Load the dataset
data = datasets.load_iris()
X = data.data
y = data.target

# Define the objective function to optimize
def svm_cv(C, gamma):
    model = SVC(C=C, gamma=gamma, random_state=42)
    return np.mean(cross_val_score(model, X, y, cv=5))

# Define the bounds of the hyperparameters
bounds = {'C': (0.1, 100), 'gamma': (0.001, 1)}

# Initialize the Bayesian optimization object
optimizer = BayesianOptimization(svm_cv, bounds)

# Streamlit app
st.title("Bayesian Optimization for Hyperparameter Search")
st.write("Optimizing hyperparameters of a Support Vector Machine (SVM) model using Bayesian optimization.")

# Sliders to vary the bounds of parameters
C_bounds = st.slider("C Bounds", 0.1, 100.0, (0.1, 100.0), step=0.1)
gamma_bounds = st.slider("Gamma Bounds", 0.001, 1.0, (0.001, 1.0), step=0.001)

# Update the bounds of the optimizer
optimizer.set_bounds({'C': C_bounds, 'gamma': gamma_bounds})

# Button to start the optimization process
if st.button("Start Optimization"):
    with st.spinner("Optimizing..."):
        optimizer.maximize(init_points=5, n_iter=25)
        best_params = optimizer.max["params"]
        st.write(f"Best parameters found: C={best_params['C']:.3f}, gamma={best_params['gamma']:.3f}")
        st.write(f"Best score: {optimizer.max['target']:.3f}")
else:
    st.write("Adjust the bounds of the hyperparameters and click the 'Start Optimization' button.")

# SL-and-UL-learning
# CS465/565 Homework 1 - Part 1: Supervised Learning

## How to run
Make sure you have python 3 installed along with the following pip packages: `pandas`, `numpy`, `scikit-learn`.  
Clone this repo then cd into `supervised-learning`  
From there, run `python3 script.py`  
The predicted values will be output to `z-predicted.csv` at program termination  

## Parameters and Architecture of the Learner
I chose to use Polynomial regression with degree 2, implemented using scikit-learn’s `PolynomialFeatures` and `Ridge` regression.  
I waas able to combine x (input attribute 1) and y (input attribute 2) into a feature matrix, then transform it into polynomial terms (1, x, y, x^2, xy, y^2) using `PolynomialFeatures(degree=2)`.  
In order to predict `z` (target value), I also chose to use Ridge regression as opposed to Linear regression. Ridge regresssion allows for more accurate learning due to the high correlation and presence of non-linearity which Linear regression wouldn't have been able to account for. I can change the `alpha=1.0` constant value for regularization to prevent overfitting.
Because of the huge scale difference, I decided to use `StandardScaler` to normalize x, y, and z (billions) for numerical stability, then unscale it for the predictions to match the original z scale.

## Justification
I chose polynomial regression with degree 2 because the relationship between x, y, and z is nonlinear, as shown by the curved, parabolic patterns in (x vs z) and (y vs z) plots.
Degree 2 captures this nonlinearity effectively with terms like x^2, xy, and y^2, while keeping the model parametric and interpretable.
With only 50 noisy samples, a higher degree (e.g., 3) risks overfitting, as it could memorize noise rather than generalize. I tested degree 3, but its tighter fit suggested overfitting, especially in sparse regions, so I prioritized degree 2 to avoid the potential risk.
Ridge regression with `alpha=1.0` prevents overfitting by penalizing large coefficients, balancing fit and complexity better than alternatives like SVR (harder to tune with small data) or neural networks (require more data).

## Trade-offs Between Model Complexity and Empirical Loss
- **Degree 2 (Lower Complexity)**: 
  - Uses 6 parameters (1, x, y, x^2, xy, y^2), reducing the risk of overfitting with 50 samples.
  - Empirical fit (visualized in `Actual vs Predicted z` plots) shows points clustering around the 45-degree line with moderate spread, indicating good generalization despite some variability in sparse regions (e.g., z values between 250e6–500e6). This spread is acceptable given the noise and limited data, and the model avoids overfitting.
- **Degree 3 (Higher Complexity)**:
  - Uses 10 parameters (1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3), capturing more nonlinearity but risking overfitting.
  - While degree 3 provided a tighter fit (less spread), it appeared to overfit in sparse regions, potentially memorizing noise rather than generalizing. I chose degree 2 to prioritize a generally robust working model over a slightly better training fit, ensuring the model performs well on unseen data with the small, noisy dataset.


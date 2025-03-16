# CS465 Machine Learning Homework (Supervised and Unsupervised)

## Part 1: Supervised Learning
### Running instructions
- This python script predicts z values from input attributes x and y using a pre-trained Polynomial Ridge Regression model.
- Here are the dependencies, copy and paste the following command if these are not already installed:
  ```bash
  pip install scikit-learn numpy pandas
- **Steps**:
  1. Make sure `x.csv`, `y.csv` files are in the root directory, as well as `trained_model_best.pkl`
  2. Run the prediction script:
     ```bash
     python3 supervised_learning.py
  3. The predicted z values from the model will be output to `z-predicted.csv`
 
### Model Architecture and Parameters
I chose to use Polynomial Ridge regression using the python scikit-learn library

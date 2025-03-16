# CS465 Machine Learning Homework (Supervised and Unsupervised)

## Part 1: Supervised Learning
### Running instructions
- This python script predicts z values from input attributes x and y using a pre-trained Polynomial Ridge Regression model.
- **Dependencies**: copy and paste the following command if these libraries are not already installed in your env:
  ```bash
  pip install scikit-learn numpy pandas
- **Steps**:
  1. Make sure `x.csv`, `y.csv` files are in the root directory, the submission zip will provide `trained_model_best.pkl` and the prediction script
  2. Run the prediction script:
     ```bash
     python3 supervised_learning.py
  3. The predicted z values from the model will be output to `z-predicted.csv`, nothing else will be output
 
### Model Architecture and Parameters
I chose to use Polynomial Ridge regression using the python scikit-learn library

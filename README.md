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
 
### Parameters and architecture
I chose to use Polynomial Ridge regression using the python scikit-learn library.

To start, I wanted to display x and y against z to see what kind of data I was handling. The csv files were 50 columns on 1 row of varying values, so I used pandas to read the csv files and then sort sort x and y into matrix 'X' as 50x2 with numpys column stack. Scikit’s regression models, including Ridge, expect the input features to be in a 2D array of shape (N samples, N features), which is why I needed to reformat the data using pandas and numpy. I created `xy.py` to simply visualize x and y against z


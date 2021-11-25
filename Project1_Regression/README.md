# Fuzzy_Systems_Regression
TSK models that are using the hybrid method for training.
For the task 1 use [
Airfoil Self-Noise Data Set](https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise).
For the task 2 use [Superconductivty Data Set](https://archive.ics.uci.edu/ml/datasets/Superconductivty+Data)
## Task 1  
Create 4 different TSK models (Models 1-4) that are using the hybrid method for training (Backpropagation and Least Squares Method)
- Use [Airfoil Self-Noise Data Set](https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise).
- Use grid partition 
- For Models 1-2 ---> Change the output function to constant and use 2 membership functions (Model 1) or 3 membership functions (Model 2)
- For Models 3-4 ---> Change the output function to linear and use 2 membership functions (Model 3) or 3 membership functions (Model 4)
- Use gbellmf as membership function type
- Train the TSK model with hybrid method (using Backpropagation and Least Squares Method)
- Evaluate the model
  - R2
  - RMSE
  - NMSE
  - NDEI
## Task 2
  - Use [Superconductivty Data Set](https://archive.ics.uci.edu/ml/datasets/Superconductivty+Data)
  - Use Subtractive Clustering (check many values of radius)
  - Use Relieff algorithm to choose the most suitable features for training 
  - Compare metrics between models to find the best parameters  with the help of grid search technique  (for the number of features and the radius used in Subtractive Clustering

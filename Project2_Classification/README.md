# Fuzzy_Systems_Classification
TSK models that are using the hybrid method for training.
For the task 1 use [Haberman's dataset](https://archive.ics.uci.edu/ml/datasets/Haberman's+Survival).
For the task 2 use [Epileptic Seizure Recognition Data Set](https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition)
## Task 1  Model 1
- Use [Haberman's dataset](https://archive.ics.uci.edu/ml/datasets/Haberman's+Survival).
- For Models 1-2, Use Subtractive Clustering (Class Independent) for clusterInfluenceRange = 0.2 (Model 1) and clusterInfluenceRange = 0.9 (Model 1) 
- For Models 3-4, Use Subtractive Clustering (Class Dependent)  for clusterInfluenceRange = 0.2 (Model 3) and clusterInfluenceRange = 0.9 (Model 4) 
- Change the output function to constant
- Train the TSK model with hybrid method (Backpropagation and Least Squares Method)
- Evaluate the model
  - Error matrix
  - Producer’s accuracy – User’s accuracy
  - Overall accuracy
  - K

## Task 2
  - Use [Epileptic Seizure Recognition Data Set](https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition)
  - Use Subtractive Clustering  (check many values of radius)
  - Use Relieff algorithm to choose the most suitable features for training 
  - Compare metrics between models to find the best parameters  with the help of grid search technique  (for the number of features and the radius used in Subtractive Clustering) and using cross validation

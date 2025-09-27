# [Project Title: Heart Disease Prediction]

## 📝 Overview

[cite_start]This project uses a dataset of patient clinical records to predict the likelihood of heart disease.The goal is to build and evaluate several classification models, fine-tune them using hyperparameter optimization, and compare their performance to identify the most effective algorithm for this task. 

---

## 💾 Dataset

The data used in this project is the "Heart Disease UCI" dataset, which contains clinical attributes from patients. [cite_start]All missing values were handled by imputing the column mean, and all features were converted to an integer data type for consistency. 

**Source:** You can find the original dataset at the UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/heart+Disease .

---

## ⚙️ Installation and Usage

To run this project locally, follow these steps:


pip install jupyter lab

**1. [cite_start]Clone the repository:** 
```bash

cd [Your Repository Folder]




# Create the environment
py -m venv venv
#if it didnt work do
python -m venv venv

# Activate on Windows
venv\Scripts\activate


bash jupyter lab \ jupyter notebook #depends on ur system , poth works for windows 

#This command will automatically open a new tab in your web browser. You will see a file manager showing the contents of your project folder.From there, click on the notebooks folder, and then click on any .ipynb file to open it and run the code cells interactively.

#if u want to run it on your device 

pip install -r requirements.txt

--------------------------------------------------------------------------------------------------------------
🛠️ Methodology
The project follows these steps:

Data Cleaning: All missing values in the dataset were imputed using the mean of their respective columns. All data types were then converted to integer. 

Preprocessing: A Scikit-Learn pipeline was used to prepare the data for modeling. Numerical features were scaled using 

StandardScaler, and categorical features were transformed using OneHotEncoder. 


Baseline Modeling: Four different classification models were trained with their default parameters to establish a baseline performance: 

Logistic Regression

Decision Tree

Random Forest

Support Vector Machine

Hyperparameter Tuning: GridSearchCV was used with 5-fold cross-validation to systematically search for the optimal set of hyperparameters for each model.


-----------------------------------------------------------------------------------------------------------------

📊 Results
After training and tuning, the models' performances were compared. The "Before Tuning" score is accuracy on a single test set, while the "After Tuning" score is the more robust average cross-validated accuracy from the training set.




to run the app u need to :

pip install streamlit

cd [Your Repository Folder]

bash streamlit run app.py





			

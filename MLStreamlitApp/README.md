## Supervised Learning via Streamlit

### Project Overview: 
This application provides an interactive walkthrough to training and evaluating a classification model, including data pre-processing, model selection, and parameter tuning. Specifically, this app explores:
- **Logistic Regression:** An algorithm which predicts class by calculating the probability of each binary outcome, as influenced by the included features.
- **'k' Nearest Neighbors:** An algorithm designed to predict class according to a data point's similarity to its 'k' nearest neighbors in the feature space.


### Usage:
To launch locally:
1. **Navigate to the proper folder:** Open your terminal and move to the directory containing 'MLStreamlitApp'  
   (e.g. cd path/to/MLStreamlitApp)
2. **Run using the following command:**  
   streamlit run Main.py
3. **Interact with the app:** Explore different datasets, supervised learning models, and tuning parameters using the app features below.
4. **Closing the app:** Close (or kill) the terminal in which the command was run to deactivate the app.

To launch via Streamlit cloud:


### App Features:
**Data Selection:** Using the sidebar, select a sample dataset or upload a CSV of your own.
**Data Processing:** Use the widgets to remove columns of choice, drop observations with missing data, and encode categorical variables.
**Model and Variable Selection:**
  - Select a supervised learning model (logistic regression or 'k' nearest neighbors)
  - Choose column to be the target variable (i.e. 'label' or 'y')
  - Specify predictor variables (i.e. features or 'X')
**Model Training and Tuning:**
  - Data will be split into training and test subsets and used for classification model of choice.
  - **Logistic Regression:**
  - **kNN:**

### References:

### Visual Examples:

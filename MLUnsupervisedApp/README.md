# Unsupervised Learning via Streamlit


## Project Overview:  


## Usage:
To launch via Streamlit cloud:
(Insert Link)

To launch locally:
1. **Navigate to the proper folder:** Open your terminal and move to the directory containing 'MLUnsupervisedApp'  
   `cd path/to/MLUnsupervisedApp`
2. **Run using the following command:**  
   `streamlit run Main.py`
4. **Interact with the app:** Explore different datasets, unsupervised learning models, and tuning parameters using the app features below.
5. **Closing the app:** Close (or kill) the terminal in which the command was run to deactivate the app.

## App Features:
**Data Selection:** Using the sidebar, select a sample dataset or upload a CSV of your own.

**Data Processing:** 
- For the sample data: Presets will automatically process the data for a uniform set-up of the sample data.
- For an uploaded CSV: Use the widgets to remove columns of choice, drop observations with missing data, and encode categorical variables.

**Model and Variable Selection:** Select an unsupervised learning model (kMeans or hierarchical clustering).
- For the sample data: Features are specified. A label is chosen and removed from the data that will be used. 
- For tan uploaded CSV: Use the instructions and provided widgets to select suitable features and a viable label, which will be removed from the data that will be used. 
    
**Model Training and Testing:**


**Hyperparameter Tuning:** Use the selectebox to choose a hyperparameter and explore its effects on the performance metrics.


## References:
**Streamlit:** [Streamlit API Cheat Sheet](https://docs.streamlit.io/develop/quick-reference/cheat-sheet), [Streamlit Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app)

## Visual Examples:


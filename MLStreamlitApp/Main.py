import streamlit as st

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ---------- TITLE ---------- #

# set up title, author, and description
st.set_page_config(layout='centered')
st.title("Supervised Learning via Streamlit:")
st.header("An interactive walkthrough data processing, model selection, and parameter tuning for classification models.")
st.write("By: Lauren Latimer | Access code on [Github](https://github.com/llatimer031/Latimer-Data-Science-Portfolio/tree/main/MLStreamlitApp)")

# ---------- FUNCTIONS ---------- #

def LogRegression(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def ClassTree(X_train, y_train):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model

def kNNClassifier(k, X_train, y_train):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model

def ConfMatrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"**Accuracy:** {accuracy:.2f}")
    
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    st.pyplot(fig)
    
# ---------- SIDEBAR ---------- #
# create sidebar to customize data and model options
st.sidebar.header("Data Options")

data_source = st.sidebar.radio("Choose a data source:",
    ("Use Sample Dataset", "Upload CSV"))
    
# upload a CSV file
if data_source == "Upload CSV":
    st.sidebar.write("**Note:** This app explores classification models, which require at least one (binary) categorical variable.")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Custom dataset uploaded successfully.")
    else:
        st.warning("Please upload a CSV file to proceed.")
        df = None

# choose from sample datasets
else:
    sample_data = st.sidebar.selectbox("Choose a sample dataset:", ['Option1', 'Option2'])
    if sample_data == 'Option1':
        df = sns.load_dataset("penguins") #placeholder data
    elif sample_data == 'Option2':
        df = sns.load_dataset("titanic") #placeholder data

if df is not None:
    # add basic file information to sidebar
    st.sidebar.subheader("File Information")
    original_dim = df.shape
    st.sidebar.write("Number of Rows:", original_dim[0])
    st.sidebar.write("Number of Columns:", original_dim[1])

    # show missing values in each column
    st.sidebar.subheader("Missing Values")
    missing_values = pd.DataFrame(df.isnull().sum(), columns=["Missing Values"]).reset_index()
    missing_values.columns = ["Column", "Missing Values"]
    st.sidebar.dataframe(missing_values)

# ---------- MAIN ---------- #

st.divider() # horizontal separator

# PART 1: DATA PROCESSING
st.header("Part 1: Processing the Data")

data_processed = False
if df is not None:
    
    # preview the chosen dataset 
    st.write("Below is a preview of your chosen dataset:")
    st.dataframe(df.head())
    
    # step 1: include only variables of choice
    st.subheader("Step 1: Filter Columns")
    st.markdown("""
    **Purpose:** Some columns may not contain helpful information, especially if many of its observations are missing values.\n
    **Action:** The sidebar shows the number of missing values in each column. Remove irrelevant variables before continuing to preserve the number of observations.
    """)
    cols = st.multiselect("Select columns to **remove**:", df.columns)
    df = df.drop(columns=cols)
    
    # step 2: remove missing values
    st.subheader("Step 2: Handle Missing Values")
    st.markdown("""
    **Purpose:** The machine learning algorithms used in this app require a dataset without missing values.\n
    **Action:** While there are several approaches to handle missing data, including dropping and imputing, this app will **drop any rows with missing values** for simplicity.
    """)
    df = df.dropna() # remove rows with missing values
    new_dim = df.shape
    incomplete_rows = original_dim[0] - new_dim[0]
    st.success(f"{incomplete_rows} incomplete observations were successfully dropped.")
    
    # step 3: encode variables
    st.subheader("Step 3: Encode Categorical Variables")
    st.markdown("""
    **Purpose:** \n
    **Action:** \n  
    **Note:** The desired target variable does not need to be encoded, as the Scikit-learn algorithms used in this app will automatically use a label encoder.         
    """)
    # create list of categorical columns to select from
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    encode_cols = st.multiselect("Select categorical columns to encode:", cat_cols)
    if encode_cols:
        df = pd.get_dummies(df, columns=encode_cols, drop_first=True)
        st.success("Selected columns were encoded using one-hot encoding.")
    else:
        st.warning("No columns selected for encoding.")

    # preview processed dataset
    # st.dataframe(df.head())
    data_processed = True # mark data as ready to continue
    
else:
    # displays if data has not been selected
    st.warning("Data is not ready for this step.")

st.divider()

# PART 2: VARIABLE SELECTION
st.header("Part 2: Model and Variable Selection") 

data_ready = False
if data_processed:
    # step 1: choose a model type to train
    st.subheader("Step 1: Choose a Classification Model")
    model_choice = st.selectbox('Select Model Type:', ['Logistic Regression', 'kNN'])

    # step 2: specify x and y columns
    st.subheader("Step 2: Choose Features and Target Variable")
    
    st.markdown("""
    **Target Variable:** 
    Please select a categorical variable. 
    Note that if you chose a logistic regression model, the target variable must also be binary.
    """)
    # select y variable
    label = st.selectbox("Select the target column (y):", df.columns)
    if label:
        # st.write(f"You have chosen the **{label}** variable to be your label.")
        y = df[label]
        st.dataframe(y.head()) 

    st.markdown("""
    **Features:** Numeric or encoded categorical variables.
    """)
    # limit feature selection to viable data types
    numeric_columns = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32', 'bool']).columns.tolist()
    if label in numeric_columns:
        numeric_columns.remove(label) # label cannot be chosen as feature
    # select x variables 
    features = st.multiselect("Select the feature columns (X):", numeric_columns)
    if features:
        # st.write(f"You have chosen the following features: **{features}**")
        X = df[features]
        st.dataframe(X.head())
    else:
        st.warning("Please select at least one feature.")
    
    
    if 'X' in locals() and 'y' in locals(): # ensures X and y have been selected
        data_ready = True # mark data as ready to continue

else:
    # displays if data has not been processed
    st.warning("Data is not ready for this step.")

st.divider()

# PART 3: TRAIN MODEL
st.header("Part 3: Train a Classification Model")
if data_ready:
    
    # step 1: split into training and test data
    st.subheader("Step 1: Train-Test Split")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    st.success("The data has been split into 80% training data and 20% test data.")
    
    # step 2: scale the data before training
    st.subheader("Step 2 (Optional): Scaling")
    st.markdown("""
    **Purpose:** For models sensitive to the scale of features, scaling can substantially improve performance. \n
    **Action:** Standardize the features (mean = 0, standard deviation = 1) and apply to train and test data. \n  
    """)
    scale_choice = st.radio("Would you like to perform scaling?", ("Yes", "No"))
    if scale_choice == 'Yes':
        scaler = StandardScaler()
        # fit the scaler on the training data and transform both training and test data
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        st.success("The data has been scaled.")
    else: 
        st.warning("Continue without scaling.")
        
    # step 3: train and test model
    st.subheader("Step 3: Train and Test the Selected Model")
    
    # run function corresponding to chosen model
    if model_choice == 'Logistic Regression':
        st.markdown("""
        **Current Model Selection:** Logistic Regression \n
        **Model Information:** \n
        """)
        
        if scale_choice == 'Yes':
            log_reg = LogRegression(X_train_scaled, y_train)
            st.markdown(""" ##### Test Results """)
            ConfMatrix(log_reg, X_test_scaled, y_test)
        else:  
            log_reg = LogRegression(X_train, y_train) 
            st.markdown(""" ##### Test Results """)
            ConfMatrix(log_reg, X_test, y_test)
        
    else:
        st.markdown(f"""
        **Current Model Selection:** 1NN \n
        **Model Information:** \n
        """)
        
        if scale_choice == 'Yes':
            knn = kNNClassifier(1, X_train, y_train)
            st.markdown(""" ##### Test Results """)
            ConfMatrix(knn, X_test_scaled, y_test)
        else:  
            knn = kNNClassifier(1, X_train, y_train)
            st.markdown(""" ##### Test Results """)
            ConfMatrix(knn, X_test, y_test)
        
    st.markdown("""
                > 💡 **Thought Question:** 
                > Does scaling the data affect the given model accuracy?
                """)
    
    # step 4: tune with different parameters
    st.subheader("Step 4: Hyperparameter Tuning")
    
    if model_choice == 'Logistic Regression':
        param = st.selectbox("Choose a parameter:", ("C", "Penalty", "Max iterations"))
    else:
        param = st.selectbox("Choose a parameter:", ("k neighbors", "Metric", "Weights"))
        
else:
    # displays if data has not been split
    st.warning("Data is not ready for this step.")

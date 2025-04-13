import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


# ---------- TITLE ---------- #
# set up title, author, and description
st.set_page_config(layout='centered')
st.title("Supervised Learning via Streamlit:")
st.header("An interactive walkthrough data processing, model selection, and parameter tuning for classification models.")
st.write("By Lauren Latimer | Access code on [Github](https://github.com/llatimer031/Latimer-Data-Science-Portfolio/tree/main/MLStreamlitApp)")

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
    st.write(f"Accuracy: {accuracy:.2f}")
    
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
    ("Upload CSV", "Use Sample Dataset"))
    
# upload a CSV file
if data_source == "Upload CSV":
    st.sidebar.write("Note: The models explored in this app work best for (binary) classification. We suggest data that contains a binary variable.")
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

# ---------- MAIN ---------- #
st.header("Part 1: Processing the Data")

data_ready = False
if df is not None:
    # preview the chosen dataset 
    st.write("Below is a preview of your chosen dataset:")
    st.dataframe(df.head())
    original_dim = df.shape
    st.write(f"The dataset contains {original_dim[0]} observations and {original_dim[1]} columns.")
    
    # include only variables of choice
    st.subheader("Step 1: Filter Columns")
    st.write("Some columns may not contain relevant information, especially if values are missing for most observations. Remove these variables before continuing to preserve the number of observations.")
    cols = st.multiselect("Select columns to keep:", df.columns)
    df = df[cols]
    
    # remove missing values
    st.subheader("Step 2: Handle Missing Values")
    st.write("Include note about options to handle missing values and choice to drop rows.") #INCOMPLETE
    df = df.dropna()
    new_dim = df.shape
    incomplete_rows = original_dim[0] - new_dim[0]
    st.success(f"{incomplete_rows} incomplete observations were successfully dropped.")
    
    # encode variables (if needed)
    st.subheader("Step 3: Encode Categorical Variables")
    st.write("For the classification models explored in this app, features must be numeric or encoded categorical variables.") # INCOMPLETE
    
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    encode_cols = st.multiselect("Select categorical columns to encode:", cat_cols)
    if encode_cols:
        df = pd.get_dummies(df, columns=encode_cols, drop_first=True)
        st.success("Selected columns encoded using one-hot encoding.")
    else:
        st.warning("No columns selected for encoding.")
        
    st.write("Note: If columns do not need to be encoded, you can skip this step.")

    st.dataframe(df.head())
    data_ready = True


st.header("Part 2: Model and Variable Selection") 

data_split = False
if data_ready:
    # choose a model to train
    st.subheader("Step 1: Choose a Classification Model")
    choice = st.selectbox('Select Model Type', ['Logistic Regression', 'Classification Tree', 'kNN'])

    # specify x and y columns
    st.subheader("Step 2: Choose Features and Target Variable")
    
    st.write("For this app, please select a target variable that is binary categorical (ex: Yes/No, True/False, 0/1).")
    label = st.selectbox("Select the target column (y):", df.columns)
    st.write(f"You have chosen the **{label}** variable to be your label.")
    y = df[label]
    st.dataframe(y.head())

    st.write("For this app, please select features that are numeric or have already been encoded.")
    # limit feature selection to numeric or encoded columns
    numeric_columns = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32', 'bool']).columns.tolist()
    # do not allow label to be chosen as feature
    if label in numeric_columns:
        numeric_columns.remove(label)
        
    features = st.multiselect("Select the feature columns (X):", numeric_columns)
    if features:
        st.write(f"You have chosen the following features: **{features}**")
        X = df[features]
        st.dataframe(X.head())
    else:
        st.warning("Please select at least one feature.")
    
    # split into training and test data
    st.subheader("Step 3: Test Train Split")
    if 'X' in locals() and 'y' in locals(): # do not execute split until x and y have been chosen
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.success("The data has been split into 80% training data and 20% test data.")
        data_split = True
    else:
        st.warning("Variable selection must be complete to continue.")
else:
    st.warning("Data is not ready for this step.")
    
st.header("Part 3: Run the Classification Model")
if data_split:
    # run the models
    if choice == 'Logistic Regression':
        log_reg = LogRegression(X_train, y_train)
        ConfMatrix(log_reg, X_test, y_test)
        
    elif choice == 'Classification Tree':
        class_tree = ClassTree(X_train, y_train)
        ConfMatrix(class_tree, X_test, y_test)
        
    else:
        k = st.slider("Select a k-value:", 1, 10)
        knn = kNNClassifier(k, X_train, y_train)
        ConfMatrix(knn, X_test, y_test)
else:
    st.warning("Data is not ready for this step.")

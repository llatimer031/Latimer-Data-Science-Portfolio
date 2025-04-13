import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


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
    print(f"Accuracy: {accuracy}")
    
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    

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
    # state dimensions
    
    # remove missing values
    st.subheader("Step 1: Handle Missing Values")
    st.write("Include note about options to handle missing values and choice to drop rows.") #INCOMPLETE
    df = df.dropna()
    st.success("Incomplete observations were successfully dropped.")
    # add note about how many rows were dropped 
    
    # encode variables (if needed)
    st.subheader("Step 2: Encode Categorical Variables")
    st.write("Explain purpose of edncoding and when to do it") # INCOMPLETE
    st.write("Note: If no columns need to be encoded, you can skip this step.")
    
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    encode_cols = st.multiselect("Select categorical columns to encode:", cat_cols)
    if encode_cols:
        df = pd.get_dummies(df, columns=encode_cols, drop_first=True)
        st.success("Selected columns encoded using one-hot encoding.")
    else:
        st.warning("No columns selected for encoding.")
        
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
    numeric_columns = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
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
        LogRegression(X_train, y_train)
    elif choice == 'Classification Tree':
        ClassTree(X_train, y_train)
    else:
        k = st.slider("Select a k-value:", 1, 10)
        kNNClassifier(k, X_train, y_train)
else:
    st.warning("Data is not ready for this step.")

import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

# ---------- TITLE ---------- #
# set up title, author, and description
st.set_page_config(layout='centered')
st.title("Supervised Learning via Streamlit:")
st.header("A Visual Approach to Model Selection and Tuning")
st.write("By Lauren Latimer | Access code on [Github](https://github.com/llatimer031/Latimer-Data-Science-Portfolio/tree/main/MLStreamlitApp)")

# ---------- FUNCTIONS ---------- #

def LogRegression():
    pass

def ClassTree():
    pass

def kNNClassifier():
    pass

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

    # specify x and y columns
    st.subheader("Step 3: Choose the Target Variable")
    label = st.selectbox("Select the target column (y):", df.columns)
    if label:
        X = df.drop(label, axis=1)
        y = df[label]
        st.write(f"You have chosen the {label} variable to be your label.")

    # split into training and test data
    st.subheader("Step 4: Test Train Split")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    
    st.header("Part 2: Train a Supervised Learning Model")
    # choose a model to train
    choice = st.selectbox('Select Model Type', ['Logistic Regression', 'Classification Tree', 'kNN'])

    if choice == 'Logistic Regression':
        LogRegression()
    elif choice == 'Classification Tree':
        ClassTree()
    else:
        kNNClassifier()
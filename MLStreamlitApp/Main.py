import streamlit as st
import pandas as pd
import seaborn as sns

# set up title, author, and description
st.set_page_config(layout='centered')
st.title("Supervised Learning via Streamlit:")
st.header("A Visual Approach to Model Selection and Tuning")
st.subheader("By Lauren Latimer | Access code on [Github](https://github.com/llatimer031/Latimer-Data-Science-Portfolio/tree/main/MLStreamlitApp)")

# create sidebar to customize data and model options
st.sidebar.header("Options")

data_source = st.sidebar.radio("Choose a data source:",
    ("Upload CSV", "Use Sample Dataset"))
    
# upload a CSV file
if data_source == "Upload CSV":
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
        df = sns.load_dataset("iris") #placeholder data

# preview the chosen dataset 
if df is not None:
    st.write("Below is a preview of your chosen dataset:")
    st.dataframe(df.head())
    
choice = st.sidebar.selectbox('Select Model Type', ['Logistic Regression', 'kNN'])

if choice == 'Logistic Regression':
    pass
elif choice == 'kNN':
    pass
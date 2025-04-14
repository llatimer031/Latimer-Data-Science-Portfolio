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

# --------------- TITLE --------------- #

# set up title, author, and description
st.set_page_config(layout='centered')
col1, mid, col2 = st.columns([10,1,30])
with col1:
    st.image("data/streamlit-app.jpeg", width=150)
with col2:
    st.title("Supervised Learning via Streamlit:")
    st.write("By: Lauren Latimer | GitHub: [llatimer031](https://github.com/llatimer031/Latimer-Data-Science-Portfolio/tree/main/MLStreamlitApp)")
    
st.header("An interactive walkthrough data processing, model selection, and parameter tuning for classification models.")

# --------------- FUNCTIONS --------------- #

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
    
# --------------- SIDEBAR --------------- #

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
    sample_data = st.sidebar.selectbox("Choose a sample dataset:", ['penguins', 'titanic'])
    if sample_data == 'penguins':
        df = sns.load_dataset("penguins") #placeholder data
    elif sample_data == 'titanic':
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

# --------------- MAIN PAGE --------------- #

st.divider() # horizontal separator

# ----- FOR CUSTOM CSV ----- #

if data_source == "Upload CSV":
    
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
        # remove rows with missing values
        df = df.dropna() 
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
    
    else: # df does not exist or is empty
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

    else: # data_processed = False
        st.warning("Data is not ready for this step.")

# ----- FOR SAMPLE DATA ----- #

else: # elif sample data set used
    
    # PART 1: DATA PROCESSING
    st.header("Part 1: Processing the Data")

    data_processed = False
    if df is not None:
    
        # preview the chosen dataset 
        st.write("Below is a preview of your chosen dataset:")
        st.dataframe(df.head())

        if sample_data == 'penguins':
            selected_cols = ['sex', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
            df = df[selected_cols]
            df = df.dropna() # remove missing values
            df = pd.get_dummies(df, columns=['sex'], drop_first=True) # encode age column
        else: # sample_data == 'titanic'
            selected_cols = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare', 'sex']
            df = df[selected_cols]
            df = df.dropna() # remove missing values
            df = pd.get_dummies(df, columns=['sex'], drop_first=True) # encode age column
            
        st.write("After removing unwanted variables, handling missing values, and encoding categorical variables, the dataset looks like:")
        st.dataframe(df.head())
        data_processed = True
        
    else: # df does not exist or is empty
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
        st.subheader("Step 2: Specify Features and Target Variable")
        
        if sample_data == 'penguins':
            features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
            X = df[features]
            y = df['sex_Male']
            st.write(f"**Target Variable:** {'sex_Male'}")
            st.write(f"**Feautres:** {features}")
            
        else: # sample_data == 'titanic'
            features = ['pclass', 'age', 'sibsp', 'parch', 'fare', 'sex_male']
            X = df[features]
            y = df['survived']
            st.write(f"**Target Variable:** {'survived'}")
            st.write(f"**Feautres:** {features}")
        
        if 'X' in locals() and 'y' in locals(): # ensures X and y have been selected
            data_ready = True # mark data as ready to continue

    else: # data_processed = False
        st.warning("Data is not ready for this step.")

st.divider()

# ----- ALL DATASETS ----- #

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
        X_train_1 = scaler.fit_transform(X_train)
        X_test_1 = scaler.transform(X_test)
        st.success("The data has been scaled.")
    else: 
        X_train_1 = X_train
        X_test_1 = X_test
        st.warning("Continue without scaling.")
        
    # step 3: train and test model
    st.subheader("Step 3: Train and Test the Selected Model")
    
    # run function corresponding to chosen model
    if model_choice == 'Logistic Regression':
        st.markdown("""
        **Current Model Selection:** Logistic Regression \n
        **Model Information:** Supervised learning model used for binary classification 
        by calculating the probability a datapoint belongs to a given class. \n
        **Coefficients and Intercept:** \n
        - Positive: Feature increases the log-odds (probability) of the target variable.
        - Negative: Feature decreases the log-odds (probability) of the target variable.
        - Intercept: Baseline of the target variable for default feature settings. 
        """)
    
        log_reg = LogRegression(X_train_1, y_train) 
        
        # Extract coefficients and intercept
        coef = pd.Series(log_reg.coef_[0], index=features)
        intercept = log_reg.intercept_[0]
        st.write(coef) # display coefficients
        st.write("\nIntercept:", intercept) # display intercept
        
        st.markdown(""" ##### Test Results """)
        ConfMatrix(log_reg, X_test_1, y_test)
        
    else:
        st.markdown(f"""
        **Current Model Selection:** 5NN \n
        **Model Information:** \n
        """)
        
        knn = kNNClassifier(5, X_train_1, y_train)
        st.markdown(""" ##### Test Results """)
        ConfMatrix(knn, X_test_1, y_test)
        
    st.markdown("""
                > 💡 **Thought Question:** 
                > Does scaling the data affect the model coefficients?
                > How about the model accuracy?
                """)
    
    # step 4: tune with different parameters
    st.subheader("Step 4: Hyperparameter Tuning")
    
    if model_choice == 'Logistic Regression':
        param = st.selectbox("Choose a parameter:", ("C", "Penalty", "Max iterations"))
    else:
        param = st.selectbox("Choose a parameter:", ("k neighbors", "Metric", "Weights"))
        if param == "k neighbors":
            st.write("Don't know what 'k' to use?")
            if st.button("Find best 'k'"): # create button to plot best k
                k_values = range(1, 11, 1)
                accuracies = []

                # Loop through different values of k, train a KNN model on scaled data, and record the accuracy for each
                for k in k_values:
                    knn_temp = KNeighborsClassifier(n_neighbors=k)
                    knn_temp.fit(X_train_1, y_train)
                    y_temp_pred = knn_temp.predict(X_test_1)
                    accuracies.append(accuracy_score(y_test, y_temp_pred))

                # Plot accuracy vs. number of neighbors (k) for the scaled data
                fig = plt.figure(figsize=(8, 5))
                plt.plot(k_values, accuracies, marker='o')
                plt.title('Accuracy vs. Number of Neighbors (k)')
                plt.xlabel('Number of Neighbors (k)')
                plt.ylabel('Accuracy')
                plt.xticks(k_values)
                st.pyplot(fig)
                
            k = st.slider("Select 'k' value:", 1, 10)
            knn_tuned = kNNClassifier(k, X_train_1, y_train)
                
        st.markdown(""" ##### Test Results """)
        ConfMatrix(knn_tuned, X_test_1, y_test)
        
else:
    # warning displays if data has not been split
    st.warning("Data is not ready for this step.")

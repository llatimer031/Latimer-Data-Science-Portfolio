# --------------- PACKAGES --------------- #

# import streamlit
import streamlit as st

# import data analysis and visualization packages
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# import machine learning packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# import unsupervised learning packages
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# --------------- TITLE --------------- #

# set up title, author, and description
st.set_page_config(layout='centered')
col1, mid, col2 = st.columns([10,1,30]) # splits the page into three columned sections
with col1: 
    # add image to first column
    st.image("https://raw.githubusercontent.com/llatimer031/Latimer-Data-Science-Portfolio/main/MLStreamlitApp/Images/streamlit-app.jpeg", width=150)
with col2:
    # add title and other info to second column
    st.title("(Un)Supervised Learning via Streamlit:")
    st.write("By: Lauren Latimer | GitHub: [llatimer031](https://github.com/llatimer031/Latimer-Data-Science-Portfolio/tree/main/MLStreamlitApp)")
    
st.header("An interactive walkthrough data processing, model selection, and parameter tuning for clustering models.")

# --------------- FUNCTIONS --------------- #

# run a k means clustering model
def graph_PCA(X_std, clusters):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_std)

    unique_clusters = np.unique(clusters)
    colors = plt.cm.get_cmap('tab10', len(unique_clusters))  # Up to 10 distinct colors

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, cluster_label in enumerate(unique_clusters):
        ax.scatter(
            X_pca[clusters == cluster_label, 0],
            X_pca[clusters == cluster_label, 1],
            color=colors(i),
            alpha=0.7,
            edgecolor='k',
            s=60,
            label=f'Cluster {cluster_label}'
        )

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.legend(loc='best')
    ax.grid(True)

    st.pyplot(fig)
    
def kCluster(k, X):   
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X)
    return kmeans, clusters

# run a hierarchical clustering model
def hier_cluster_graph(X, y, method):
    Z = linkage(X, method=method)
    
    labels = y.to_list()

    plt.figure(figsize=(20, 7))
    dendrogram(Z, labels = labels)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.ylabel("Distance")
    plt.show()
    
def elbow_plot(k_range, X):
    wcss = []
    
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X)
        wcss.append(km.inertia_)  # inertia: sum of squared distances within clusters
        
    # plot the result
    plt.figure(figsize=(12, 5))
    plt.plot(k_range, wcss, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.show()
    
def sil_plot_kmeans(k_range, X):
    silhouette_scores = [] 
    
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_std)
        labels = km.labels_
        silhouette_scores.append(silhouette_score(X, labels))
    
    # plot the result
    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette_scores, marker='o', color='green')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal k')
    plt.grid(True)
    plt.show()  
    
    best_k = k_range[np.argmax(silhouette_scores)]
    return best_k
        
def sil_plot_hier(k_range, X):
    silhouette_scores = [] 
    
    for k in k_range:
        # Fit hierarchical clustering with Ward linkage (same as dendrogram)
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X_scaled)

        # Silhouette: +1 = dense & well‑separated, 0 = overlapping, −1 = wrong clustering
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)

    # Plot the curve
    plt.figure(figsize=(7,4))
    plt.plot(list(k_range), silhouette_scores, marker="o")
    plt.xticks(list(k_range))
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Average Silhouette Score")
    plt.title("Silhouette Analysis for Agglomerative (Ward) Clustering")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    best_k = k_range[np.argmax(silhouette_scores)]
    return best_k

# --------------- SIDEBAR --------------- #

# create sidebar to customize data and model options
st.sidebar.header("Data Options")

# allow the use to choose a to use a sample dataset or upload a CSV
data_source = st.sidebar.radio("Choose a data source:",
    ("Use Sample Dataset", "Upload CSV"))
    
# option (a) upload a CSV file
if data_source == "Upload CSV":
    st.sidebar.write("**Note:** Add any relevant notes")
    
    # provide space to upload a CSV file of choice
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None: # if a file has been uploaded
        # use pandas to read CSV into a python dataframe
        df = pd.read_csv(uploaded_file) 
        st.success("Custom dataset uploaded successfully.")
    else: # if a file has not been uploaded
        st.warning("Please upload a CSV file to proceed.")
        # set python dataframe to None
        df = None 

# option (b) choose from sample datasets
else:
    # allow users to select which sample dataset they want to use
    sample_data = st.sidebar.selectbox("Choose a sample dataset:", ['penguins', 'titanic'])
    
    if sample_data == 'penguins':
        # load Seaborn's penguins dataset and assign to python dataframe
        df = sns.load_dataset("penguins") 
    elif sample_data == 'titanic':
        # load Seaborn's titanic dataset and assign to python dataframe
        df = sns.load_dataset("titanic")

# runs if data is in the df, whether it be from a CSV or sample data
if df is not None:
    # add basic file information to sidebar
    st.sidebar.subheader("File Information")
    original_dim = df.shape # gets dimensions of original df
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

# for an uploaded csv, allow the user to perform custom processing and variable selection
if data_source == "Upload CSV":
    
    # PART 1: DATA PROCESSING
    st.header("Part 1: Processing the Data")

    data_processed = False # initialize as false to check when processing is complete
    if df is not None:
    
        # preview the first few rows of the chosen dataset 
        st.write("Below is a preview of your chosen dataset:")
        st.dataframe(df.head())
    
        # step 1: include only variables of choice
        st.subheader("Step 1: Filter Columns")
        # explain purpose of removing irrelevant or unhelpful variables
        st.markdown("""
        **Purpose:** Some columns may not contain helpful information, especially if many of its observations are missing values.\n
        **Action:** The sidebar shows the number of missing values in each column. Remove irrelevant variables before continuing to preserve the number of observations.
        """)
        # allow user to select from a list of the df's columns
        cols = st.multiselect("Select columns to **remove**:", df.columns)
        df = df.drop(columns=cols) # drop the selected columns
    
        # step 2: remove missing values
        st.subheader("Step 2: Handle Missing Values")
        # explain purpose of removing missing values
        st.markdown("""
        **Purpose:** The machine learning algorithms used in this app require a dataset without missing values.\n
        **Action:** While there are several approaches to handle missing data, 
        including dropping or imputing values, this app will **drop any rows with missing values** for simplicity.
        """)
        df = df.dropna() # remove rows with missing values
        new_dim = df.shape # get dimensions of new df, after removing incomplete rows
        incomplete_rows = original_dim[0] - new_dim[0] # calculate how many rows were removed
        st.success(f"{incomplete_rows} incomplete observations were successfully dropped.")
    
        # step 3: encode variables
        st.subheader("Step 3: Encode Categorical Variables")
        # explain purpose of encoding variables
        st.markdown("""
        **Purpose:** These clustering algorithms require numerical inputs, 
        which inhibits them from interpreting categorical variables in their natural state.\n
        **Action:** If you wish to use categorical data as a feature, convert the variable into numeric form using one-hot encoding. \n  
        """)
        # create list of categorical columns to select from
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        # allow user to select multiple of the categorical columns to encode
        encode_cols = st.multiselect("Select categorical columns to encode:", cat_cols)
        
        if encode_cols: # if the user selected columns
            # one-hot encode variables and save to existing df
            df = pd.get_dummies(df, columns=encode_cols, drop_first=True)
            st.success("Selected columns were encoded using one-hot encoding.")
        else: # if the user has not selected any columns
            st.warning("No columns selected for encoding.")

        data_processed = True # mark data as ready to continue
    
    else: # df does not exist or is empty
        st.warning("Data is not ready for this step.")

    st.divider()

    # PART 2: VARIABLE SELECTION
    st.header("Part 2: Variable Selection") 

    data_ready = False # initialize as false to check when data is ready to continue to model training
    
    if data_processed: # checks if data has been successfully pre-processed in last step

        # step 2: specify column to use as label 
        st.subheader("Step 1: Remove Labels")
        
        st.markdown("""
        **Label:** Choose a column suitable to act as a label. This column will not be used in the clustering algorithm. \n
        **Note:** While unsupervised methods do not use labels directly,
        this choice can be used later to check the accuracy of the clusters produced. 
        """)
        # allow user to select a label among the columns
        label = st.selectbox("Select a label to **exclude**:", df.columns)
        if label: # checks that a y variable has been chosen by user
            y = df[label] # creates variable y by sub setting column from df
            st.dataframe(y.head()) # preview the first few rows of the y variable

        # step 3: specify columns to use as features
        st.subheader("Step 2: Choose Features")
        
        st.markdown("""
        **Features:** Please select numeric or encoded categorical variables for the model to use during clustering. 
        """)
        # limit feature selection to viable data types
        numeric_columns = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32', 'bool']).columns.tolist() # extracts numeric cols
        if label in numeric_columns:
            numeric_columns.remove(label) # ensures that label cannot be chosen as feature
            
        # allow user to select features from the numeric columns
        features = st.multiselect("Select features to **include**:", numeric_columns)
        if features: # ensures feature columns have been selected by user
            X = df[features] # create df X by subsetting the selected feature columns from the main df
            st.dataframe(X.head()) # displays first few rows of X df
        else: # no feature variables have been selected
            st.warning("Please select at least one feature.")
        
        if 'X' in locals() and 'y' in locals(): # ensures X and y variables have been selected
            data_ready = True # mark data as ready to continue

    else: # data_processed = False
        st.warning("Data is not ready for this step.")

# ----- FOR SAMPLE DATA ----- #

# for a sample dataset, the app processes the data and selected variables automatically
else: # elif sample data set used (not custom CSV)
    
    # PART 1: DATA PROCESSING
    st.header("Part 1: Processing the Data")

    data_processed = False # initialize as false to check when pre processing is complete
    
    if df is not None:
    
        # preview the first few rows of the chosen dataset 
        st.write("Below is a preview of your chosen dataset:")
        st.dataframe(df.head())

        if sample_data == 'penguins':
            selected_cols = ['sex', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
            df = df[selected_cols] # subset the data by the selected columns
            df = df.dropna() # remove missing values
            df = pd.get_dummies(df, columns=['sex'], drop_first=True) # encode sex column
            
        else: # sample_data == 'titanic'
            selected_cols = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare', 'sex']
            df = df[selected_cols] # subset the data by selected columns
            df = df.dropna() # remove missing values
            df = pd.get_dummies(df, columns=['sex'], drop_first=True) # encode age column
            
        st.write("After removing unwanted variables, handling missing values, and encoding necessary categorical variables, the dataset looks like:")
        st.dataframe(df.head()) # preview the dataset after processing
        
        data_processed = True # mark data as processed 
        
    else: # df does not exist or is empty
        st.warning("Data is not ready for this step.")

    st.divider()

    # PART 2: VARIABLE SELECTION
    st.header("Part 2: Variable Selection") 

    data_ready = False # initialize as false to check when data is ready to continue to modeling stages
    
    if data_processed: # checks that data has been successfully processed
        
        # step 1: specify features and label
        st.subheader("Step 1: Remove Label and Specify Features")
        
        if sample_data == 'penguins':
            features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
            X = df[features] # create X df by subsetting feature columns from original df
            y = df['sex_Male'] # create Y df by subsetting y col from original df
            
            # preview selected variables
            st.write(f"**Label:** {'sex_Male'}")
            st.write(f"**Feautres:** {features}")
            st.write("**Note:** The label column will *not* be used during the clustering algorithm itself, but it may be used later to check the accuracy of the clusters produced.")
            
        else: # sample_data == 'titanic'
            features = ['pclass', 'age', 'sibsp', 'parch', 'fare', 'sex_male']
            X = df[features] # create X df by subsetting feature columns from original df
            y = df['survived'] # create Y df by subsetting y col from original df
            
            # preview selected variables
            st.write(f"**Label:** {'survived'}")
            st.write(f"**Feautres:** {features}")
            st.write("**Note:** The label column will *not* be used during the clustering algorithm itself, but it may be used later to check the accuracy of the clusters produced.")
        
        if 'X' in locals() and 'y' in locals(): # ensures X and y have been selected
            data_ready = True # mark data as ready to continue

    else: # data_processed = False
        st.warning("Data is not ready for this step.")

st.divider()

# ----- ALL DATASETS ----- #

# PART 3: TRAIN MODEL
st.header("Part 3: Fit a Clustering Model")

if data_ready: # checks if data is ready from previous steps
    
    st.subheader("Step 1: Choose a Model Type")
    # allow user to choose between model types
    model_choice = st.selectbox('Select Model Type:', ['kMeans', 'hierarchical'])
        
    # step 2: scale the data before training
    st.subheader("Step 2: Scaling")
    st.markdown("""
    **Purpose:** Clustering algorithms rely on distance metrics, which are sensitive to the scale of data. \n
    **Action:** Standardize the features (mean = 0, standard deviation = 1) and apply to the data \n  
    """)
    
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    
    if X_std is not None:
        st.success("The data has been successfully scaled.")
        
    # step 2: compute clusters
    st.subheader("Step 3: Compute Clusters")
    
    # compute clusters corresponding to chosen model
    if model_choice == 'kMeans':
        st.markdown("""
        **Current Model Selection:** kMeans Clustering \n
        **Model Information:**  \n
        """)
        
        # initialize by choosing a k to start
        
        st.write("**i) Initialization:** *k* initial centroids will randomly be chosen.")
        # allow user to select a value of k to use
        k = st.slider("Please select a value for *k*:", 1, 10)
        
        # run the kMeans algorithm to fit the model
        
        st.markdown("""
        **ii) Fitting the Model:**
        - Each point will be assigned to the nearest centroid, based on the distance metric  
        - The centroids will be recalculated using the mean of each cluster  
        - Repeat until stopping criteria is met
        """)

        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(X_std)
        
        if clusters is not None:
            st.success("kMeans model has been succesfully fit to the data.")
        
    else: # model selection is hierarchical
        st.markdown(f"""
        **Current Model Selection:** Hierarchical Clustering \n
        **Model Information:** 
        """)
    
    # visualize the results using PCA
        
    st.subheader("Step 4: Visualize the Results using PCA")
    graph_PCA(X_std, clusters)

    st.divider()

    # PART 4: HYPERPARAMETER TUNING
    st.header("Part 4: Hyperparameter Tuning")
    
    st.write("**Overview:** Fine tuning a model helps to find optimal hyperparameters to maximize accuracy and other performance indicators.")
    
    if model_choice == 'Logistic Regression':
        # explain hyperparameter options
        st.markdown("""
        **Options:**
        - **penalty:** Penalties are a form of regularization, in which the model is simplified to avoid overfitting. 
            - `'l2'` (default): Ridge regression shrinks the weights of all coefficients, but does not eliminate any.
            - `'l1'`: Lasso regression limits the weight of some coefficients to 0, reducing the size of the model.
            - `None`: No penalty is added.
        - **max_iter:** Maximum number of iterations taken for the solver to converge (default = 100).
        """)
        
        st.subheader("Step 1: Choose a Hyperparameter")
        param = st.selectbox("Select a parameter to explore:", ("penalty", "max_iter"))
        
        if param == "penalty": # penalty is selected
            penalty = st.selectbox("Choose a penalty to apply:", ("l2", "l1", "None"))
            
            # intialize model according to selected penalty
            if penalty == "l2":
                log_reg_tuned = LogisticRegression(penalty=penalty) # create model with l2 penalty
            elif penalty == "l1":
                log_reg_tuned = LogisticRegression(penalty='l1', solver='liblinear') # specify solver that supports l1
            else:
                log_reg_tuned = LogisticRegression(penalty=None) # create model with no penalty
                
            log_reg_tuned.fit(X_train_1, y_train) # train model on data
            
        else: # max_iter is selected
            max_iter = st.number_input("Input a number of iterations to run:", 1, 10000)
            log_reg_tuned = LogisticRegression(max_iter=max_iter) # create model with specified iterations
            log_reg_tuned.fit(X_train_1, y_train) # train model on data
        
        st.subheader("Step 2: Analyze Performance")
        
        # extract coefficients and intercept
        coef1, coef2 = st.columns(2) # create columns to put tables side-by-side
        
        # set default and tuned parameter choices for clarity
        if param == "penalty":
            default = 'l2'
            tuned_choice = penalty
        else: 
            default = '100 iterations'
            tuned_choice = f'{max_iter} iterations'
        
        # input coefficients into columns
        with coef1:
            st.write(f"**Model Coefficients before Tuning ({default}):**")
            coef = pd.Series(log_reg.coef_[0], index=features)
            intercept = log_reg.intercept_[0]
            st.write(coef) # display coefficients
            st.write("\nIntercept:", intercept) # display intercept
        with coef2:
            st.write(f"**Model Coefficients after Tuning ({tuned_choice}):**")
            coef2 = pd.Series(log_reg_tuned.coef_[0], index=features)
            intercept2 = log_reg_tuned.intercept_[0]
            st.write(coef2) # display coefficients
            st.write("\nIntercept:", intercept2) # display intercept
        
        # thought questions about accuracy / conf matrix
        if param == "penalty":
            st.markdown("""
                > 💭 **Thought Question:** 
                > How do the model coefficients change when penalties are added? 
                > Does the 'l1' penalty eliminate any coefficients completely? 
                """)
        
        st.markdown("#### Test Data Results")   
        ConfMatrix(log_reg_tuned, X_test_1, y_test) # create conf matrix for tuned model
        
        # thought questions about accuracy / conf matrix
        if param == "max_iter":
            st.markdown("""
                > 💭 **Thought Question:** 
                > Is there a threshold in which increasing the number of iterations no longer changes the outcomes? \n
                > 💡 **Analysis:** 
                > When max_iter is high enough, the model will converge before the manual stopping point is reached. 
                > If model accuracy is still increasing as you increase max_iter, then the model is likely being stopped before full convergence.
                """)
            
        else: # add question about penalty
            st.markdown("""
                > 💭 **Thought Question:** 
                > Does penalizing the coefficients increase or decrease model accuracy? \n
                > 💡 **Analysis:**
                > If adding penalties increase the model performance, the non-penalized model may be overfit. 
                > If the addition of penalties does not increase model performance, then overfitting may not be a major concern for the model. 
                """)
        
    else: # model_choice == 'kNN'
        # explain hyperparameter options
        st.markdown("""
        **Options:**
        - **n_neighbors:** Specifies 'k', or the number of neighbors to compare each test point to (default = 5)).
        - **metric:** Specifies the metric used to calculate distance between data points. 
            - `minkowski` (default): standard euchlidean (straight-line) distance
            - `manhattan`: grid-like distance between coordinates
            - `chebyshev`: maximum absolute difference
        """)
        
        st.subheader("Step 1: Choose a Hyperparameter")
        param = st.selectbox("Select a parameter to explore:", ("n_neighbors", "metric"))
        
        if param == "n_neighbors":
            k = st.slider("Select 'k' value to use:", 1, 20)
            knn_tuned = kNNClassifier(k, X_train_1, y_train) # create model with selected k
            
            # add option to find best k
            if st.toggle("Not sure what 'k' to use? Click here to find best 'k'"): # create button to plot best k
                k_values = range(1, 21, 1)
                accuracies = []

                # loop through different values of k, train a KNN model on data, and record the accuracy for each
                for k in k_values:
                    knn_temp = KNeighborsClassifier(n_neighbors=k)
                    knn_temp.fit(X_train_1, y_train)
                    y_temp_pred = knn_temp.predict(X_test_1)
                    accuracies.append(accuracy_score(y_test, y_temp_pred))

                # plot accuracy vs. number of neighbors (k) for the data
                fig = plt.figure(figsize=(8, 5))
                plt.plot(k_values, accuracies, marker='o')
                plt.title('Accuracy vs. Number of Neighbors (k)')
                plt.xlabel('Number of Neighbors (k)')
                plt.ylabel('Accuracy')
                plt.xticks(k_values)
                st.pyplot(fig)
                
        else: # if param == "metric"
            chosen_metric = st.selectbox("Select a metric:", ('minkowski', 'manhattan', 'chebyshev'))
            knn_tuned = KNeighborsClassifier(metric=chosen_metric) # create model with the selected metric
            knn_tuned.fit(X_train_1, y_train) # train new model on data
            
                
        st.subheader("Step 2: Analyze Performance")
        ConfMatrix(knn_tuned, X_test_1, y_test) # create conf matrix for tuned model
        
        # thought questions / analysis
        if param == "metric":
            st.markdown("""
                > 💭 **Thought Question:** 
                > Compare the outcomes for different metrics using both scaled and unscaled data.
                > Why might these metrics have a smaller affect on scaled data? \n
                > 💡 **Analysis:** Scaling the data sets equal variances to each feature, 
                > so no single feature will outweigh the others in distance calculations.
                """)
        else: # add question about k
            st.markdown("""
                > 💭 **Thought Question:** Why is it important to choose an optimal 'k'? \n
                > 💡 **Analysis:**  Too small of a k may be sensitive to noise and lead to overfitting, 
                > whereas too large of a k may overgeneralize and cause underfitting.
                """)
        
else:
    # warning displays if data has not been split
    st.warning("Data is not ready for this step.")

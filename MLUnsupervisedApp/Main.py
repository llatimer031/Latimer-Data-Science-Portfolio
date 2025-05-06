# --------------- PACKAGES --------------- #

# import streamlit
import streamlit as st

# import data analysis and visualization packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# import machine learning packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_wine

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
    sample_data = st.sidebar.selectbox("Choose a sample dataset:", ['wine', 'breast cancer'])
    
    if sample_data == 'wine':
        # load the wine dataset from sklearn
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)  
        df['target'] = data.target  # Target variable (cultivar class)
        feature_names = data.feature_names
        target_names = data.target_names
        
    elif sample_data == 'breast cancer':
        # load the breast cancer dataset from sklearn
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)  
        df['target'] = data.target  # Target variable (diagnosis)
        feature_names = data.feature_names
        target_names = data.target_names

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

        st.markdown("""
        The selected dataset is already pre-processed:  
        - The data does not contain missing values.  
        - All features are numeric.
        """)
        
        st.success("Data is ready to move on to the next step.")
        
        data_processed = True # mark data as processed 
        
    else: # df does not exist or is empty
        st.warning("Data is not ready for this step.")

    st.divider()

    # PART 2: VARIABLE SELECTION
    st.header("Part 2: Variable Selection") 

    data_ready = False # initialize as false to check when data is ready to continue to modeling stages
    
    if data_processed: # checks that data has been successfully processed
        
        # step 1: specify features and label
        st.subheader("Step 1: Specify Label and Features")
        
        X = df.drop(columns=['target'])  # all features except the label
        y = df['target']  # target column

        if sample_data == 'wine':
            st.write(f"**Label:** Cultivar Class")
        else: # if sample_data == 'breast cancer'
            st.write(f"**Label:** Diagnosis")
            
        st.write(f"**Features:** {list(X.columns)}")
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
        
    # step 3: compute clusters
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

        kmeans = KMeans(n_clusters=k, random_state=42) # run kMeans algorithm
        clusters = kmeans.fit_predict(X_std) # assigns predictions to clusters variable
        
        # displays success message if model has been run
        if clusters is not None:
            st.success("The kMeans model has been succesfully fit to the data.")
        
    else: # model selection is hierarchical
        st.markdown(f"""
        **Current Model Selection:** Hierarchical Clustering \n
        **Model Information:** 
        """)
        
        # build a dendrogram
        st.write("**i) Building a Hierarchical Tree:** Merge clusters until complete.")
        
        Z = linkage(X_std) # will create linkage matrix with default linkage method "ward"

        labels = y.to_list() # y is the label selected earlier

        # plot dendrogram in streamlit
        fig, ax = plt.subplots(figsize=(20, 7))
        dendrogram(Z, truncate_mode="lastp", labels=labels, ax=ax) # creates dendrogram with limited examples shown
        ax.set_title("Hierarchical Clustering Dendrogram")
        ax.set_ylabel("Distance")

        st.pyplot(fig)
        
        # assign k number of clusters using the dendrogram
        st.write("**ii) Choose the Number of Clusters:** Inspect the dendrogram to choose an appropriate number of *k* clusters.")
        # allow user to select k based on inspect
        k = st.slider("Please select a value for *k*:", 1, 10)
        
        # run the agglomerative clustering algorithm to fit the model
        st.markdown("""
        **ii) Fitting the Model:**
        Agglomerative clustering with the same linkage method will produce integer labels for the dataframe.
        """)
        
        agg = AgglomerativeClustering(n_clusters=k) # default linkage is ward
        clusters = agg.fit_predict(X_std) # save the predictions to the cluster variable
        
        # display success message if model has been run
        if clusters is not None:
            st.success("The hierarchical model has been succesfully fit to the data.")
            
    # step 4: visualize the results using PCA
        
    st.subheader("Step 4: Visualize Clusters using Principle Component Analysis (PCA)")
    st.markdown("""
        **Purpose:** When data is high-dimensional, it can become difficult to both analyze and interpret.
        This event is called the *curse of dimensionality,* a problem in which PCA aims to solve
        by reducing the data to a specified number of dimensions.\n
        **Action:** Combine features into principle components that capture maximum variance in the data.
        """)
    graph_PCA(X_std, clusters)
    
    if st.toggle("Need help interpreting these principle components? Click here for explanation."):
        st.write("The first principle component...")
        # COMPLETE ANALYSIS
        
    st.markdown("""
                > 💭 **Thought Question:** 
                Are the clusters created by this unsupervised learning algorithm well separated?
                """)

    # step 5: analyze model performance
    st.subheader("Step 5: Analyze Model Performance")
    
    # import accuracy score to calculate the percentage of data points correctly predicted
    accuracy = accuracy_score(y, clusters)
    st.write(f"Accuracy Score: {accuracy * 100:.2f}%")
    
    st.divider()

    # PART 4: HYPERPARAMETER TUNING
    st.header("Part 4: Hyperparameter Tuning")
    
    st.write("**Overview:** Fine tuning a model helps to find optimal hyperparameters to maximize accuracy and other performance indicators.")
    
    if model_choice == 'kMeans':
        # explain hyperparameter options
        st.markdown("""
        **Options:**
        - **n_clusters:** 
        - **max_iter:**
        """)
        
        st.subheader("Step 1: Choose a Hyperparameter")
        # allow user to select a parameter from the given options
        param = st.selectbox("Select a parameter to explore:", ("n_clusters", "max_iter"))
        
        if param == "k": # penalty is selected
            # explain elbow plot
            # plot elbow plot
            
            # explain silhouette plot
            # plot silhouette plot
            pass
            
        else: # max_iter is selected
            max_iter = st.number_input("Input a number of iterations to run:", 1, 10000)
            kmeans_tuned = KMeans(n_clusters=k, max_iter=max_iter, random_state=42) # create model with specified iterations
            clusters_tuned = kmeans_tuned.fit_predict(X_std) # fit model to data to create clusters
        
        st.subheader("Step 2: Analyze Performance")
        
        
        
    else: # model_choice == 'hierarchical'
        # explain hyperparameter options
        st.markdown("""
        **Options:**
        - **linkage:**
        - **n_clusters:**
        """)
        
        st.subheader("Step 1: Choose a Hyperparameter")
        param = st.selectbox("Select a parameter to explore:", ("linkage", "n_clusters"))
        
        if param == "n_clusters":
            pass
            # explain the silhouette plot
            # plot silhouette scores
                
        else: # if param == "linkage"
            linkage_type = st.selectbox("Select a linkage type", ('ward', 'option2'))
            # create model with the selected linkage
            # fit model to data to create clusters
            
                
        st.subheader("Step 2: Analyze Performance")
        
else:
    # warning displays if data has not been split
    st.warning("Data is not ready for this step.")

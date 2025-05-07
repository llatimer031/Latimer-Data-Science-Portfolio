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
st.set_page_config(layout='wide')
col1, col2 = st.columns([1,6]) # splits the page into three columned sections
with col1: 
    # add image to first column
    st.image("https://raw.githubusercontent.com/llatimer031/Latimer-Data-Science-Portfolio/main/MLStreamlitApp/Images/streamlit-app.jpeg", width=175)
with col2:
    # add title and other info to second column
    st.write("") # vertical space for formatting
    st.write("") # vertical space for formatting
    st.write("") # vertical space for formatting
    st.title("(Un)Supervised Learning via Streamlit:")
    st.write("By: Lauren Latimer | GitHub: [llatimer031](https://github.com/llatimer031/Latimer-Data-Science-Portfolio/tree/main/MLStreamlitApp)")
    
st.header("An interactive walkthrough data processing, model selection, and parameter tuning for clustering models.")

# --------------- FUNCTIONS --------------- #

# function that will map clusters onto 2D space using PCA
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

    ax.set_title('Cluster Mapping: 2D PCA')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.legend(loc='best')
    ax.grid(True)

    st.pyplot(fig) #display plot

# function used to find the elbow point in WCSS scores across k-values 
def elbow_plot(k_range, X_std):
    wcss = [] # initialize empty list to store WCSS values
    
    for k in k_range: # iterate over the given range of k values
        km = KMeans(n_clusters=k, random_state=42) # run model on given number of clusters
        km.fit(X_std) # fit model to data
        # append to list
        wcss.append(km.inertia_)  # inertia: sum of squared distances within clusters
        
    # plot the result in streamlit
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_range, wcss, marker='o')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Within-Cluster Sum of Squares (WCSS)')
    ax.set_title('Elbow Method for Optimal k')
    ax.grid(True)

    st.pyplot(fig) #display plot

# function used to plot kMeans silhouette scores across k-values 
def sil_plot_kmeans(k_range, X_std):
    silhouette_scores = []

    k_range = [k for k in k_range if k > 1]  # excludes k=1 since silhouette score is undefined

    silhouette_scores = [] # initialize empty list to store scores
    for k in k_range: # iterate through provided k range (excluding k=1)
        km = KMeans(n_clusters=k, random_state=42) # run model on given number of clusters
        labels = km.fit_predict(X_std) # fit to data
        # use silhouette score package to obtain score
        score = silhouette_score(X_std, labels)
        # append to list
        silhouette_scores.append(score) 

    # store the best k by finding the max silhouette score
    best_k = k_range[np.argmax(silhouette_scores)]

    # plot the results in streamlit
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_range, silhouette_scores, marker='o', color='green')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Score for Optimal k')
    ax.grid(True)
    ax.legend()

    st.pyplot(fig) #display plot
    
    # display success message for best k
    st.success(f"The best 'k' based on silhouette score is **{best_k}**.")
    return best_k

# function used to plot hierarchical silhouette scores across k-scores 
def sil_plot_hier(k_range, X_std, linkage='ward'):
    silhouette_scores = []  # initialize empty list to store scores
    
    k_range = [k for k in k_range if k > 1] # exclude k=1 because silhouette score is undefined
    for k in k_range:
        # Fit hierarchical clustering with Ward linkage (same as dendrogram)
        labels = AgglomerativeClustering(n_clusters=k, linkage=linkage).fit_predict(X_std)

        # Silhouette: +1 = dense & well‑separated, 0 = overlapping, −1 = wrong clustering
        score = silhouette_score(X_std, labels)
        silhouette_scores.append(score)
        
    # store the best k after finding max score
    best_k = k_range[np.argmax(silhouette_scores)]

    # plot the results in streamlit
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_range, silhouette_scores, marker='o', color='green')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Score for Optimal k')
    ax.grid(True)
    ax.legend()

    st.pyplot(fig) #display plot
    
    # display success message for best k
    st.success(f"The best 'k' based on silhouette score is **{best_k}**.")
    return best_k

# create function to display message corresponding to accuracy score
def accuracy_message(accuracy):
    if accuracy >= 0.75:
        # displays green success message for high accuracy
        st.success(f"{accuracy * 100:.0f}% of the predicted cluster labels matched the true labels.")
    elif accuracy >= 0.50:
        # displays yellow 'warning' message for moderate accuracy
        st.warning(f"{accuracy * 100:.0f}% of the predicted cluster labels matched the true label.")
    else:
        # displays red error message for poor accuracy
        st.error(f"{accuracy * 100:.0f}% of the predicted cluster labels matched the true label.")
        if st.toggle("Low accuracy score? Click here for some potential reasons why."):
            st.markdown("""
            - **Data:** The data may not have distinct clusters.
            - **Incorrect 'k':** The model is using a different number of clusters than found in the true label. 
            - **Label Mismatch:** The cluster labels created by the model are arbitrary, and may not match the true labels.
                - For example, if you have a binary label and an accuracy score of 0%, that indicates that each of the observations are correctly grouped, but the labels are simply swapped. 
            """)
# --------------- SIDEBAR --------------- #

# create sidebar to customize data and model options
st.sidebar.header("Data Options")

# allow the use to choose a to use a sample dataset or upload a CSV
data_source = st.sidebar.radio("Choose a data source:",
    ("Use Sample Dataset", "Upload CSV"))
    
# option (a) upload a CSV file
if data_source == "Upload CSV":
    st.sidebar.markdown("""
    **Note:** While unsupervised learning algorithms do not use labeled data to build the model itself,
    this app utilizes labels to evaluate performance. \n
    \n 
    **Recommendation:** Upload a dataset with a **binary** label
    in order to most effectively explore this app's performance measures. 
    """)
    
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
        
        # add note about extra processing for csv
        st.markdown("""
        <div style="background-color: #f5f5f5; padding: 15px; border-radius: 6px; border: 1px solid #ccc">
        <b>Note:</b> When using your own CSV file, this app will require a few additional steps 
        to ensure that your data is ready for modeling. 
        </div>
        """, unsafe_allow_html=True)
        st.write("") # vertical space
    
        # preview the first few rows of the chosen dataset 
        st.write("Below is a preview of your chosen dataset:")
        st.dataframe(df.head())
    
        # step 1: include only variables of choice
        st.subheader("Step 1: Filter Columns")
        # explain purpose of removing irrelevant or unhelpful variables
        st.markdown("""
        **Purpose:** Some columns may not contain helpful information, especially if many of its observations are missing values.\n
        **Action:** The sidebar shows the number of missing values in each column. Remove irrelevant variables before continuing to preserve the number of observations during the next step.
        """)
        
        # allow user to select from a list of the df's columns
        cols = st.multiselect("Select columns to **remove**:", df.columns)
        df = df.drop(columns=cols) # drop the selected columns
        
        # display status message
        if cols:
            st.success(f"The following columns have been succesfully removed: {cols}")
        else:
            st.warning("No columns were selected for removal. Select a column *or* proceed without filtering.")
    
        # step 2: remove missing values
        st.subheader("Step 2: Handle Missing Values")
        # explain purpose of removing missing values
        st.markdown("""
        **Purpose:** The unsupervised learning algorithms used in this app require a dataset without missing values.\n
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
        **Purpose:** Unsupervised learning algorithms require numerical inputs, 
        which inhibits them from interpreting categorical variables in their natural state.\n
        **Action:** If you wish to use categorical data as a feature, convert the variable into numeric form using one-hot encoding. \n  
        """)
        
        # create list of categorical columns to select from
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # check to see if the dataset contained categorical columns
        if cat_cols:
            # allow user to select multiple of the categorical columns to encode
            encode_cols = st.multiselect("Select categorical columns to encode:", cat_cols)
            
            if encode_cols: # if the user selected columns
                # one-hot encode variables and save to existing df
                df = pd.get_dummies(df, columns=encode_cols, drop_first=True)
                st.success("Selected columns were encoded using one-hot encoding.")
            else: # if the user has not selected any columns
                st.warning("No columns selected for encoding.")
                
        else: # no categorical columns
            st.success("This dataset contains no categorical columns. Proceed without encoding.")
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
        Binary labels are recommended for optimal compatibility with the performance metrics in this app.
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

# PART 3: FIT MODEL
st.header("Part 3: Fit a Clustering Model")

if data_ready: # checks if data is ready from previous steps
    
    st.subheader("Step 1: Choose a Model Type")
    # allow user to choose between model types
    model_choice = st.selectbox('Select Model Type:', ['kMeans', 'hierarchical'])
        
    # step 2: scale the data before training
    st.subheader("Step 2: Scaling")
    st.markdown("""
    **Purpose:** Clustering algorithms rely on distance metrics, which are sensitive to the scale of data. \n
    **Action:** Standardize the features (mean = 0, standard deviation = 1) and apply to the data. \n  
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
        **Model Information:** This method is an algorithm which groups unlabeled data into 'k' clusters 
        by finding the optimal centroid for each grouping.\n
        """)
        
        # initialize by choosing a k to start
        
        st.write("**i) Initialization:** 'k' initial centroids will randomly be chosen.")
        # allow user to select a value of k to use
        k = st.slider("Please select a value for 'k':", 1, 10, value=2)
        
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
        **Current Model Selection:** (Agglomerative) Hierarchical Clustering \n
        **Model Information:** This method is an algorithm in which each observation is treated as its own cluster 
        and then merged with the nearest cluster until complete (or told to stop).
        """)
        
        # i.) build a dendrogram
        st.write("**i) Building a Hierarchical Tree:** Merge clusters until only one remains using ward linkage [(more info on linkage types here)](#step-1-find-the-best-linkage-method).")
        
        Z = linkage(X_std, method="ward") # will create linkage matrix using ward linkage
        labels = y.to_list() # y is the label selected earlier

        # plot dendrogram in streamlit
        fig, ax = plt.subplots(figsize=(20, 7))
        dendrogram(Z, truncate_mode="lastp", labels=labels, ax=ax) # creates dendrogram with limited examples shown
        ax.set_title("Hierarchical Clustering Dendrogram (Ward Linkage)")
        ax.set_ylabel("Distance")
        ax.set_xlabel("Lables")
        st.pyplot(fig) # display figure
        
        # ii.) assign k number of clusters using the dendrogram
        st.markdown("""
        **ii) Choose the Number of Clusters:** 
        Inspect the dendrogram to choose an appropriate number of 'k' clusters.
        This should be at the point where merge distances show a clear distinction.
        """)
        # allow user to select k based on inspect
        k = st.slider("Please select a value for 'k':", 1, 10, value=2)
        
        st.markdown("""
        <div style="background-color: #f5f5f5; padding: 15px; border-radius: 6px; border: 1px solid #ccc">
        💡 <b>Hint:</b> If you are still unsure of what 'k' to use based on the dendrogram,  
        the silhouette plot in <a href="#step-2-find-the-optimal-number-of-clusters">part 4.2</a> may help guide your decision.
        </div>
        """, unsafe_allow_html=True)
        st.write("") # add vertical space after box

        
        # iii) run the agglomerative clustering algorithm to fit the model
        st.markdown("""
        **iii) Fitting the Model:**
        Agglomerative clustering with the same linkage method will produce integer labels for the dataframe.
        """)
        
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward") 
        clusters = agg.fit_predict(X_std) # save the predictions to the cluster variable
        
        # display success message if model has been run
        if clusters is not None:
            st.success("The hierarchical model has been successfully fit to the data.")
            
    # step 4: analyze results
        
    st.subheader("Step 4: Analyze Model Performance")
    
    # look at visual representation of clusters on PCA graph
    st.write("**i) Visualize Clusters using Principle Component Analysis (PCA)**")
    
    st.markdown("""
        **Purpose:** When data is high-dimensional, it can become difficult to both analyze and interpret.
        This event is called the *curse of dimensionality,* a problem in which PCA aims to solve
        by reducing the data to a specified number of dimensions.\n
        **Action:** Combine features into principle components that capture maximum variance in the data.
        """)
    # use defined function to graph clusters on principle components
    col1, col2 = st.columns(2)
    with col1:
        # add centered label using markdown
        st.markdown("<h4 style='text-align: center;'>Predicted Clusters</h4>", unsafe_allow_html=True)
        graph_PCA(X_std, clusters)
    with col2:
        st.markdown("<h4 style='text-align: center;'>True Labels</h4>", unsafe_allow_html=True)
        graph_PCA(X_std, y)
    
    # allow option for user to interpret the principle components
    if st.toggle("Need help interpreting these principle components? Click here for explanation."):
        st.write("The first principle component...")
    
    st.markdown("""
    <div style="background-color: #f5f5f5; padding: 15px; border-radius: 6px; border: 1px solid #ccc">
    💭 <b>Thought Question:</b> Are the clusters created by this unsupervised learning algorithm well separated?
    </div>
    """, unsafe_allow_html=True)  
    st.write("") # vertical space following box  

    # step 5: analyze model performance
    st.write("**ii) Calculate Accuracy Scores**")
    
    # import accuracy score to calculate the percentage of data points correctly predicted
    accuracy = accuracy_score(y, clusters)
    st.write(f"Accuracy Score: {accuracy:.2f}")
    accuracy_message(accuracy)
    
    st.divider()

    # PART 4: HYPERPARAMETER TUNING
    st.header("Part 4: Hyperparameter Tuning")
    
    st.write("**Overview:** Fine tuning a model helps to find optimal hyperparameters to maximize accuracy and other performance indicators.")
    
    if model_choice == 'kMeans':
        # explain hyperparameter options
        st.markdown("""
        **Options:**
        - **n_clusters (k):** specifies the number of initial centroids (and thus number of clusters) that the model creates.
        - **max_iter:** maximum number of iterations for the k-means algorithm.
        """)
        
        st.subheader("Step 1: Find the Optimal Number of Clusters")
        
        st.write("**Purpose:** Testing multiple 'k' values allows us to find the optimal number of clusters for the data.")
        # allow user to pick a range of k-values to explore
        min_k, max_k = st.slider("Select a range of 'k' values to test:", 1, 20, (1,10), step=1)
        k_range = range(min_k, max_k + 1)
        
        # allow user to pick which option they want to use to find best k
        st.write("**Options:** For kMeans clustering, elbow and silhouette plots are frequently used to find the best 'k' value.")
        option = st.selectbox("Select a method to proceed", ("Elbow", "Silhouette"))
        
        if option == "Elbow":
            # explain the use of elbow plots
            st.markdown("""
            **Option 1: Elbow Method** \n
            Elbow plots track the within-cluster sum of squares (WCSS) against different 'k' values.
            The 'elbow' point (the point at which the rate of decrease suddenly changes) demonstrates an optimal 'k' value. \n
            """)
            elbow_plot(k_range, X_std)
            # ask user to set the best k by visually inspecting the plot
            best_k = st.number_input("Visually inspect the plot then enter the best 'k':", min_k, max_k)
            # display success message for best k
            st.success(f"The best 'k' based on the WCSS is **{best_k}.**")
            
        else: # option == "Silhouette"
            # explain the use of silhouette scores
            st.markdown("""
            **Option 2: Silhouette Score** \n
            Silhouette scores measure how similar an observation is to its own cluster compared to other clusters,
            in which a higher score indicates better fit. 
            A silhouette plot calculates the average silhouette score of all observations
            and tracks this average across different 'k'.
            """)
            best_k = sil_plot_kmeans(k_range, X_std)
        
        st.markdown("""
        <div style="background-color: #f5f5f5; padding: 15px; border-radius: 6px; border: 1px solid #ccc">
        💭 <b>Thought Question:</b> Do both methods suggest the same 'k' value?
        </div>
        """, unsafe_allow_html=True)  
        st.write("") # verticle space following box 
        
        # step 2: set max_iter to find the convergence point
        st.subheader("Step 2: Investigate Convergence")
        # allow user to input the number of iterations to run
        max_iter = st.number_input("Input a number of iterations to run:", 1, 10000)

        # run the model with the adjusted number of iterations
        kmeans_tuned = KMeans(n_clusters=best_k, max_iter=max_iter, random_state=42) # create model with specified iterations
        clusters_tuned = kmeans_tuned.fit_predict(X_std) # fit model to data to create clusters
        
        # visualize PCA results
        graph_PCA(X_std, clusters_tuned)
        st.markdown("""
        <div style="background-color: #f5f5f5; padding: 15px; border-radius: 6px; border: 1px solid #ccc">
        💭 <b>Thought Question:</b> Are you able to identify any observations switching clusters between iterations?
        </div>
        """, unsafe_allow_html=True)  
        st.write("") # verticle space following box 
        
        # step 3: analyze performance
        st.subheader("Step 3: Analyze Performance")
        st.markdown("""
        ##### Performance Metrics
        """)
        
        st.write("**i) Calculate an updated accuracy score**")
        # calculate accuracy
        accuracy_tuned = accuracy_score(y, clusters_tuned)
        st.write(f"Accuracy Score: {accuracy_tuned:.2f}")
        
        st. write("**ii) Calculate an updated silhouette score**")
        # calculate silhouette score
        silhouette_tuned = silhouette_score(X_std, clusters_tuned)
        st.write(f"Silhouette Score: {silhouette_tuned: .2f}")
        
        st.markdown("""
        <div style="background-color: #f5f5f5; padding: 15px; border-radius: 6px; border: 1px solid #ccc">
        💭 <b>Thought Question:</b> Did making parameter adjustments improve the performance metrics from your initial model? 
        Which had a larger impact, number of clusters or number of iterations?
        </div>
        """, unsafe_allow_html=True)  
        st.write("") # verticle space following box 
    
        
    else: # model_choice == 'hierarchical'
        # explain hyperparameter options
        st.markdown("""
        **Options:**
        - **linkage:** specifies the calculation performed to identify next 'nearest' cluster.
        - **n_clusters (k):** specifies the number of clusters the algorithm creates.
        """)
        
        # step 1: test alternate linkages
        st.subheader("Step 1: Find the Best Linkage Method")
        st.markdown("""
        **Linkage Options:**
        - **Ward:** Minimizes variance within clusters (used above). Operates most similar to kMeans.
        - **Complete:** Uses the maximum distance between any two observations in two clusters.
        - **Single:** Uses the minimum distance between any two observations in two clusters. 
        - **Average:** Uses the average of all pairwise distances between clusters. This is sometimes referred to as 'chaining' for its long clusters. 
        """)
        
        # allow user to select linkage type
        linkage_type = st.selectbox("Select a linkage type:", ("ward", "complete", "single", "average"))
        
        # create linkage matrix using chosen linkage methods
        Z_tuned = linkage(X_std, method=linkage_type) 
        
        st.write(f"See the dendrogram for {linkage_type} linkage below:")
        # plot dendrogram in streamlit
        fig, ax = plt.subplots(figsize=(20, 7))
        dendrogram(Z_tuned, truncate_mode="lastp", labels=labels, ax=ax) # creates dendrogram with limited examples shown
        ax.set_title(f"Hierarchical Clustering Dendrogram ({linkage_type})")
        ax.set_ylabel("Distance")
        ax.set_xlabel("Labels")
        st.pyplot(fig) # display figure
        
        # step 2: find the optimal number of clusters
        st.subheader("Step 2: Find the Optimal Number of Clusters")
        
        st.write("**Purpose:** Testing multiple 'k' values allows us to find the optimal number of clusters for the data.")
        # allow user to pick a range of k-values to explore
        min_k, max_k = st.slider("Select a range of 'k' values to test:", 1, 20, (1,10), step=1)
        k_range = range(min_k, max_k + 1)
        
        st.markdown("""
        **Approach: Silhouette Score** \n
        Silhouette scores measure how similar an observation is to its own cluster compared to other clusters,
        in which a higher score indicates better fit. 
        A silhouette plot calculates the average silhouette score of all observations
        and tracks this average across different 'k'.
        """)
        # use function to display plot and return best k
        best_k = sil_plot_hier(k_range, X_std, linkage=linkage_type) 
        
        # step 3: analyze performance
        st.subheader("Step 3: Analyze Performance")
        # run agglomerative clustering using chosen linkage and best_k
        agg_tuned = AgglomerativeClustering(n_clusters=best_k, linkage=linkage_type) 
        clusters_tuned = agg_tuned.fit_predict(X_std) # save the predictions to the cluster variable
        
        st.write("With the chosen parameters, the predicted clusters look like:")
        # visualize PCA results
        graph_PCA(X_std, clusters_tuned)
        
        st.markdown("""
        ##### Performance Metrics
        """)
        
        st. write("**i) Calculate an updated accuracy score**")
        # calculate accuracy
        accuracy_tuned = accuracy_score(y, clusters_tuned)
        st.write(f"Accuracy Score: {accuracy_tuned:.2f}")
        
        st. write("**ii) Calculate an updated silhouette score**")
        # calculate silhouette score
        silhouette_tuned = silhouette_score(X_std, clusters_tuned)
        st.write(f"Silhouette Score: {silhouette_tuned: .2f}")
        
        st.markdown("""
        <div style="background-color: #f5f5f5; padding: 15px; border-radius: 6px; border: 1px solid #ccc">
        💭 <b>Thought Question:</b> Which combination of linkage and 'k' yields the best performance values? 
        Did performance improve from the initial model?
        </div>
        """, unsafe_allow_html=True)  
        st.write("") # verticle space following box 
    
        
else:
    # warning displays if data has not been split
    st.warning("Data is not ready for this step.")

# Unsupervised Learning via Streamlit


## Project Overview:  
In the [MLStreamlitApp](https://github.com/llatimer031/Latimer-Data-Science-Portfolio/edit/main/MLStreamlitApp), we explored supervised learning techniques on a variety of datasets. However, not all data has a clear target variable, or in other words, not all data is 'labeled'. Consequently, this app provides an interactive walkthrough to fitting and evaluating *unsupervised* learning models, including:

- **kMeans Clustering:** An algorithm which groups unlabeled data into 'k' clusters by finding the optimal centroid for each grouping. 
- **(Agglomerative) Hierarchical Clustering:** An algorithm in which each observation is treated as its own cluster and then merged with the nearest cluster until told to stop.

in which hidden structures are revealed without the use of labels. Both methods are commonly used and proven effective, though hierarchical clustering may provide a better starting point if you do not know an initial number of clusters to begin with.

## Usage:
To launch via Streamlit cloud:
(Insert Link)

To launch locally:
1. **Clone the repository:**
   `git clone repository_url`
2. **Navigate to the proper folder:** Open your terminal and move to the directory containing 'MLUnsupervisedApp'  
   `cd path/to/MLUnsupervisedApp`
3. **Run using the following command:**  
   `streamlit run Main.py`
4. **Interact with the app:** Explore different datasets, unsupervised learning models, and tuning parameters using the app features below.
5. **Closing the app:** Close (or kill) the terminal in which the command was run to deactivate the app.

## App Features:
**Data Selection:** Using the sidebar, select a sample dataset or upload a CSV of your own.

**Data Processing:** 
- For the sample data: The datasets are fully processed upon download. 
- For an uploaded CSV: Use the widgets to remove unwanted columns, drop observations with missing data, and encode categorical variables.

**Variable Selection:** 
- For the sample data: Label and features are specified. Note that the label will not be used to fit the model. 
- For tan uploaded CSV: Use the instructions and provided widgets to select suitable features and a viable label, which will be removed from the data before fitting the model. 
    
**Model Training and Testing:** Select an unsupervised learning model (kMeans or hierarchical clustering).
- Scale the data to ensure accurate performance.
- Fit the data with a specified number of 'k' clusters.
  - For hierarchical clustering, build a dendrogram to guide this decision.
- Analyze performace by visualizing clusters (PCA mapping) and computing accuracy.

**Hyperparameter Tuning:** Explore each hyperparameter and evaluate its effects on the performance metrics.
- kMeans:
  - Use the selectbox to view a method (elbow or silhouette) for evaluating the best 'k'.
  - Input a number to select the maximum number of iterations run.
- Hierarchical:
  - Use the selectbox to explore four linkage types (ward, complete, single, average).
  - Calculate silhouette scores for a range of 'k' to determine the optimal value.


## References:
**Data:** [SciKit-Learn load_wine](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html), [SciKit-Learn load_breast_cancer](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)

**Streamlit:** [Streamlit API Cheat Sheet](https://docs.streamlit.io/develop/quick-reference/cheat-sheet), [Streamlit Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app)

**kMeans:** [SkiKit-Learn kMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)

**Hierarchical:** [SciKit-Learn Agglomerative](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html), [SciPy Linkage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html), [SciPy Dendrogram](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html)

## Visual Examples:


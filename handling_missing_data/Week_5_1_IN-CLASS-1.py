import pandas as pd            # Library for data manipulation
import seaborn as sns          # Library for statistical plotting
import matplotlib.pyplot as plt  # For creating custom plots
import streamlit as st         # Framework for building interactive web apps

# ================================================================================
#Missing Data & Data Quality Checks
#
# This lecture covers:
# - Data Validation: Checking data types, missing values, and ensuring consistency.
# - Missing Data Handling: Options to drop or impute missing data.
# - Visualization: Using heatmaps and histograms to explore data distribution.
# ================================================================================
st.title("Missing Data & Data Quality Checks")
st.markdown("""
This lecture covers:
- **Data Validation:** Checking data types, missing values, and basic consistency.
- **Missing Data Handling:** Options to drop or impute missing data.
- **Visualization:** Using heatmaps and histograms to understand data distribution.
""")

# ------------------------------------------------------------------------------
# Load the Dataset
# ------------------------------------------------------------------------------
# Read the Titanic dataset from a CSV file.
df = pd.read_csv('data/titanic.csv')

# ------------------------------------------------------------------------------
# Display Summary Statistics
# ------------------------------------------------------------------------------
# Show key statistical measures like mean, standard deviation, etc.
st.write("**Summary Statistics**")
st.dataframe(df.describe())

# ------------------------------------------------------------------------------
# Check for Missing Values
# ------------------------------------------------------------------------------
# Display the count of missing values for each column.
st.write("**Number of Missing Values by Column**")
# Compute the sum of missing values per column and display the result.
st.dataframe(df.isnull().sum())

# ------------------------------------------------------------------------------
# Visualize Missing Data
st.write("**Heatmap of Missing Values**")

# ------------------------------------------------------------------------------
# Create a heatmap to visually indicate where missing values occur.
# Create a matplotlib figure and axis for the heatmap.
fig, ax = plt.subplots()
# Plot a heatmap where missing values are highlighted (using the 'viridis' color map, without a color bar).
sns.heatmap(df.isnull(), cmap="viridis", cbar=False)
# Render the heatmap in the Streamlit app.
st.pyplot(fig)

# ================================================================================
# Interactive Missing Data Handling
#
# Users can select a numeric column and choose a method to address missing values.
# Options include:
# - Keeping the data unchanged
# - Dropping rows with missing values
# - Dropping columns if more than 50% of the values are missing
# - Imputing missing values with mean, median, or zero
# ================================================================================
st.subheader("Handle Missing Data")

# Work on a copy of the DataFrame so the original data remains unchanged.
column = st.selectbox("Choose a column to fill:", df.select_dtypes(include=['number']).columns)

# Apply the selected method to handle missing data.
method = st.radio("Choose a method:",
                  ["Original DF", "Drop Rows", "Drop Columns",
                   "Impute Mean", "Impute Median", "Impute Zero"])

# Copy our original data frame
# df is going to remain untouched, df_clean is going to be our imputation/deletion df
df_clean = df.copy()

if method == "Original DF":
    pass
elif method == "Drop Rows":
    # drop any row with a missing value
    df_clean = df_clean.dropna()
elif method == "Drop Columns":
    # drop only the columns where more than 50% of the data is missing
    df_clean = df_clean.drop(columns=df_clean.columns[df_clean.isnull().mean() > 0.5])
elif method == "Impute Mean":
    # replace missing values with the mean of that column
    df_clean[column] = df_clean[column].fillna(df[column].mean())
elif method == "Impute Median":
    # replace missing values with the median of that column
    df_clean[column] = df_clean[column].fillna(df[column].median())
elif method == "Impute Zero":
    # replace missing values with zero
    df_clean[column] = df_clean[column].fillna(0)
    
st.write(df_clean.describe())

st.subheader("Cleaned Dta Distribution")
fig, ax = plt.subplots()
sns.histplot(df_clean[column], kde = True)
st.pyplot(fig)
# ------------------------------------------------------------------------------
# Compare Data Distributions: Original vs. Cleaned
#
# Display side-by-side histograms and statistical summaries for the selected column.
# ------------------------------------------------------------------------------


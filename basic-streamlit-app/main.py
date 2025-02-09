import streamlit as st
import pandas as pd

# import the data
df = pd.read_csv('data/penguins.csv')

st.title("Palmer's Penguins Data")
st.header("This app will allow you to explore different characteristics of a dataset of penguins, including location, species, and size traits.")

# filter the data by a selected island
island = st.selectbox("First, choose an island to explore:", df["island"].unique())
island_df = df[df["island"] == island]

# report how many species are on the island
num_species = len(island_df["species"].unique())
st.subheader(f"There are {num_species} specie(s) of penguins on {island} island.")

# allow user to sort by species
sort_species = st.radio("Would you like to sort by species?", ("No", "Yes"))

if sort_species == "Yes":
    species = st.selectbox("Please select a species:", island_df["species"].unique())
    filtered_df = island_df[island_df["species"] == species] 
    st.subheader(f"You are now viewing {species} penguins on {island} island.")
else:
    filtered_df = island_df  
    st.subheader(f"You are viewing all species on {island} island.")

# ask user to select a column to explore
selected_col = st.selectbox("Choose a size characteristic to sort by:", 
                         ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"])

# create a slider for the chosen column
size = st.slider("Choose a range for the selected characteristic:", 
                 min_value = filtered_df[selected_col].min(), 
                 max_value = filtered_df[selected_col].max())

# display text according to chosen column
if selected_col == "bill_length_mm":
    st.subheader(f"Penguins with bill length under {size} mm:")
elif selected_col == "bill_depth_mm":
    st.subheader(f"Penguins with bill depth under {size} mm:")
elif selected_col == "flipper_length_mm":
    st.subheader(f"Penguins with flipper length under {size} mm:")
else:
    st.subheader(f"Penguins with body mass under {size} g:")
    
# display subset of data
filtered_df2 = filtered_df[filtered_df[selected_col] <= size]
st.dataframe(filtered_df2)
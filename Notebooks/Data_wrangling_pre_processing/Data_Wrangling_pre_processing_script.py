#!/usr/bin/env python
# coding: utf-8

# <img src="https://github.com/akthammomani/akthammomani/assets/67468718/8d1f93b4-2270-477b-bd76-f9ec1c075307" width="1700"/>

# # Data Wrangling: AI-Powered Heart Disease Risk Assessment
# 
# * **Name:** Aktham Almomani
# * **Course:** Probability and Statistics for Artificial Intelligence (MS-AAI-500-02) / University Of San Diego
# * **Semester:** Summer 2024
# * **Group:** 8
# 
# ![data_wrangling](https://github.com/akthammomani/AI_powered_heart_disease_risk_assessment_app/assets/67468718/3d94f886-b57a-46e0-aa55-65c036add226)

# ## **Contents**<a is='Contents'></a>
# * [Introduction](#Introduction)
# * [Dataset](#Dataset)
# * [Setup and Preliminaries](#Setup_and_preliminaries)
#   * [Import Libraries](#Import_libraries)
#   * [Necessary Functions](#Necessary_Functions)
# * [Extracting descriptive column names for the dataset](#Extracting_descriptive_column_names_for_the_dataset)
# * [Importing dataset](#Importing_dataset)
# * [Validating the dataset](#Validating_the_dataset)
# * [Correcting dataset column names](#Correcting_dataset_column_names)
# * [Heart Disease related features](#Heart_Disease_related_features)
# * [Selection Heart disease related features](#Selection_Heart_disease_related_features)
# * [Imputing Missing Data, Transforming Columns and Features Engineering](#Imputing_missing_Data_and_transforming_columns)
#   * [Distribution-Based Imputation](#Distribution_Based_Imputation)
#   * [Column 1: Are_you_male_or_female](#Column_1_Are_you_male_or_female)
#   * [Column 2: Ever_Diagnosed_with_Angina_or_Coronary_Heart_Disease](#Column_2_Ever_Diagnosed_with_Angina_or_Coronary_Heart_Disease)
#   * [Column 3: Computed_race_groups_used_for_internet_prevalence_tables](#Column_3_Computed_race_groups_used_for_internet_prevalence_tables)
#   * [Column 4: Imputed_Age_value_collapsed_above_80](#Column_4_Imputed_Age_value_collapsed_above_80)
#   * [Column 5: General_Health](#Column_5_General_Health)
#   * [Column 6: Have_Personal_Health_Care_Provider](#Column_6_Have_Personal_Health_Care_Provider)
#   * [Column 7: Could_Not_Afford_To_See_Doctor](#Column_7_Could_Not_Afford_To_See_Doctor)
#   * [Column 8: Length_of_time_since_last_routine_checkup](#Column_8_Length_of_time_since_last_routine_checkup)
#   * [Column 9: Ever_Diagnosed_with_Heart_Attack](#Column_9_Ever_Diagnosed_with_Heart_Attack)
#   * [Column 10: Ever_Diagnosed_with_a_Stroke](#Column_10_Ever_Diagnosed_with_a_Stroke)
#   * [Column 11: Ever_told_you_had_a_depressive_disorder](#Column_11_Ever_told_you_had_a_depressive_disorder)
#   * [Column 12: Ever_told_you_have_kidney_disease](#Column_12_Ever_told_you_have_kidney_disease)
#   * [Column 13: Ever_told_you_had_diabetes](#Column_13_Ever_told_you_had_diabetes)
#   * [Column 14: Computed_body_mass_index_categories](#Column_14_Computed_body_mass_index_categories)
#   * [Column 15: Difficulty_Walking_or_Climbing_Stairs](#Column_15_Difficulty_Walking_or_Climbing_Stairs)
#   * [Column 16: Computed_Physical_Health_Status](#Column_16_Computed_Physical_Health_Status)
#   * [Column 17: Computed_Mental_Health_Status](#Column_17_Computed_Mental_Health_Status)
#   * [Column 18: Computed_Asthma_Status](#Column_18_Computed_Asthma_Status)	
#   * [Column 19: Exercise_in_Past_30_Days](#Column_19_Exercise_in_Past_30_Days)
#   * [Column 20: Computed_Smoking_Status](#Column_20_Computed_Smoking_Status)
#   * [Column 21: Binge_Drinking_Calculated_Variable](#Column_21_Binge_Drinking_Calculated_Variable)	
#   * [Column 22: How_Much_Time_Do_You_Sleep](#Column_22_How_Much_Time_Do_You_Sleep)	
#   * [Column 23: Computed_number_of_drinks_of_alcohol_beverages_per_week](#Column_23_Computed_number_of_drinks_of_alcohol_beverages_per_week)
# * [Dropping unnecessary columns](#Dropping_unnecessary_columns)
# * [Review final structure of the cleaned dataframe](#Review_final_structure_of_the_cleaned_dataframe)
# * [Saving the cleaned dataframe](#Saving_the_cleaned_dataframe)

# ## **Introduction**<a id='Introduction'></a>
# [Contents](#Contents)
# 
# In this notebook, I have undertaken a series of data wrangling steps to prepare our dataset for analysis. **Data wrangling** is a crucial step in the data science process, involving the transformation and mapping of raw data into a more usable format. Here's a summary of the key steps taken in this notebook:
# 
# * **Dealing with Missing Data:** Identified and imputed missing values in critical columns, such as the gender column, ensuring the dataset's completeness.
# * **Data Mapping:** Transformed categorical variables into more meaningful representations, making the data easier to analyze and interpret.
# * **Data Cleaning:** Removed or corrected inconsistent and erroneous entries to improve data quality.
# * **Feature Engineering:** Created new features that may enhance the predictive power of our models.
# These steps are essential for building a reliable and robust model for heart disease prediction.

# ## **Dataset**<a id='Dataset'></a>
# [Contents](#Contents)
# 
# * The Behavioral Risk Factor Surveillance System (BRFSS) is the nation’s premier system of health-related telephone surveys that collect state data about U.S. residents regarding their health-related risk behaviors, chronic health conditions, and use of preventive services. Established in 1984 with 15 states, BRFSS now collects data in all 50 states as well as the District of Columbia and three U.S. territories. CDC BRFSS completes more than 400,000 adult interviews each year, making it the largest continuously conducted health survey system in the world.
# * The dataset was sourced from Kaggle [(Behavioral Risk Factor Surveillance System (BRFSS) 2022)](https://www.kaggle.com/datasets/ariaxiong/behavioral-risk-factor-surveillance-system-2022/data) and it was originally downloaded from the [CDC BRFSS 2022 website.](https://www.cdc.gov/brfss/annual_data/annual_2022.html)
# * To get more understanding regarding the dataset, please go to the [data_directory](https://github.com/akthammomani/AI_powered_health_risk_assessment_app/tree/main/data_directory) folder in my [Github](https://github.com/akthammomani).

# ## **Setup and preliminaries**<a id='Setup_and_preliminaries'></a>
# [Contents](#Contents)

# ### Import libraries<a id='Import_libraries'></a>
# [Contents](#Contents)

# In[1]:


#Let's import the necessary packages:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import scipy.stats as stats
from scipy.stats import gamma, linregress
from bs4 import BeautifulSoup
import re
from fancyimpute import KNN
import dask.dataframe as dd

# let's run below to customize notebook display:
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# format floating-point numbers to 2 decimal places: we'll adjust below requirement as needed for specific answers during this assignment:
pd.set_option('float_format', '{:.2f}'.format)


# ### **Necessary  functions**<a id='Necessary_Functions'></a>
# [Contents](#Contents)

# In[2]:


def summarize_df(df):
    """
    Generate a summary DataFrame for an input DataFrame.   
    Parameters:
    df (pd.DataFrame): The DataFrame to summarize.
    Returns:
    A datafram: containing the following columns:
              - 'unique_count': No. unique values in each column.
              - 'data_types': Data types of each column.
              - 'missing_counts': No. of missing (NaN) values in each column.
              - 'missing_percentage': Percentage of missing values in each column.
    """
    # No. of unique values for each column:
    unique_counts = df.nunique()    
    # Data types of each column:
    data_types = df.dtypes    
    # No. of missing (NaN) values in each column:
    missing_counts = df.isnull().sum()    
    # Percentage of missing values in each column:
    missing_percentage = 100 * df.isnull().mean()    
    # Concatenate the above metrics:
    summary_df = pd.concat([unique_counts, data_types, missing_counts, missing_percentage], axis=1)    
    # Rename the columns for better readibility
    summary_df.columns = ['unique_count', 'data_types', 'missing_counts', 'missing_percentage']   
    # Return summary df
    return summary_df
#-----------------------------------------------------------------------------------------------------------------#
# Function to clean and format the label
def clean_label(label):
    # Replace any non-alphabetic or non-numeric characters with nothing
    label = re.sub(r'[^a-zA-Z0-9\s]', '', label)
    # Replace spaces with underscores
    label = re.sub(r'\s+', '_', label)
    return label
#-----------------------------------------------------------------------------------------------------------------#

# Function to impute missing values based on distribution
def impute_missing(row):
    if pd.isna(row['Are_you_male_or_female_3']):
        return np.random.choice(value_counts.index, p=value_counts.values)
    else:
        return row['Are_you_male_or_female_3']


#-----------------------------------------------------------------------------------------------------------------#
def value_counts_with_percentage(df, column_name):
    # Calculate value counts
    counts = df[column_name].value_counts(dropna=False)
    
    # Calculate percentages
    percentages = df[column_name].value_counts(dropna=False, normalize=True) * 100
    
    # Combine counts and percentages into a DataFrame
    result = pd.DataFrame({
        'Count': counts,
        'Percentage': percentages
    })
    
    return result


# ## **Extracting descriptive column Names for the dataset**<a id='Extracting_descriptive_column_names_for_the_dataset'></a>
# [Contents](#Contents)
# 
# The Behavioral Risk Factor Surveillance System (BRFSS) dataset available on Kaggle, found here, contains a wealth of information collected through surveys. However, the column names in the dataset are represented by short labels or codes (e.g., _STATE, FMONTH, IDATE), which can be difficult to interpret without additional context.
# 
# To ensure we fully understand what each column in the dataset represents, it is crucial to replace these short codes with their corresponding descriptive names. These descriptive names provide clear insights into the type of data each column holds, making the dataset easier to understand and analyze.
# 
# **Process Overview:**
# * **Identify the Source for Descriptive Names:** The descriptive names corresponding to these short labels are typically documented in the [codebook in HTML](https://github.com/akthammomani/AI_powered_health_risk_assessment_app/tree/main/data_directory) or metadata provided by the data collection authority. In this case, the descriptive names are found in an HTML document provided by the BRFSS.
# * **Parse the HTML Document:** Using web scraping techniques, such as BeautifulSoup in Python, we can parse the HTML document to extract the relevant information. Specifically, we look for tables or sections in the HTML that list the short labels alongside their descriptive names.
# * **Match and Replace:** We create a mapping of short labels to their descriptive names. This mapping is then applied to our dataset to replace the short labels with more meaningful descriptive names.
# * **Save the Enhanced Dataset:** The dataset with descriptive column names is saved for subsequent analysis, ensuring that all users can easily interpret the columns.

# In[3]:


# Path to the HTML file:
file_path = 'USCODE22_LLCP_102523.HTML'

# Read the HTML file:
with open(file_path, 'r', encoding='windows-1252') as file:
    html_content = file.read()

# Parse the HTML content using BeautifulSoup:
soup = BeautifulSoup(html_content, 'html.parser')

# Find all the tables that contain the required information:
tables = soup.find_all('table', class_='table')

# Initialize lists to store the extracted data:
labels = []
sas_variable_names = []

# Loop through each table to extract 'Label' and 'SAS Variable Name':
for table in tables:
    # Find all 'td' elements in the table:
    cells = table.find_all('td', class_='l m linecontent')
    
    # Loop through each cell to find 'Label' and 'SAS Variable Name':
    for cell in cells:
        text = cell.get_text(separator="\n")
        label = None
        sas_variable_name = None
        for line in text.split('\n'):
            if line.strip().startswith('Label:'):
                label = line.split('Label:')[1].strip()
            elif line.strip().startswith('SAS\xa0Variable\xa0Name:'):
                sas_variable_name = line.split('SAS\xa0Variable\xa0Name:')[1].strip()
        if label and sas_variable_name:
            labels.append(label)
            sas_variable_names.append(sas_variable_name)
        else:
            print("Label or SAS Variable Name not found in the text:")
            print(text)

# Create a DataFrame:
data = {'SAS Variable Name': sas_variable_names, 'Label': labels}
cols_df = pd.DataFrame(data)

# Save the DataFrame to a CSV file:
output_file_path = 'extracted_data.csv'
cols_df.to_csv(output_file_path, index=False)

print(f"Data has been successfully extracted and saved to {output_file_path}")

cols_df.head()


# In[4]:


#let's run below to examin each features again missing data count & percentage, unique count, data types:
summarize_df(cols_df)


# No Missing Data - looks like we have 324 columns 

# ## **Importing dataset**<a id='Importing_dataset'></a>
# [Contents](#Contents)

# In[5]:


#First, let's load the main dataset BRFSS 2022:
df = pd.read_csv('brfss2022.csv')


# ## **Validating the dataset**<a id='Validating_the_dataset'></a>
# [Contents](#Contents)

# In[6]:


# Now, let's look at the top 5 rows of the df:
df.head()


# In[7]:


#now, let's look at the shape of df:
shape = df.shape
print("Number of rows:", shape[0], "\nNumber of columns:", shape[1])


# ## **Correcting dataset column names**<a id='Correcting_dataset_column_names'></a>
# [Contents](#Contents)
# 
# To replace the SAS Variable Names in your dataset with the corresponding labels (where spaces in the labels are replaced with underscores), you can follow these steps:
# 
# * Create a mapping from the SAS Variable Names to the modified labels.
# * Use this mapping to rename the columns in your dataset.
# 

# In[8]:


# Function to clean and format the label
def clean_label(label):
    # Replace any non-alphabetic or non-numeric characters with nothing
    label = re.sub(r'[^a-zA-Z0-9\s]', '', label)
    # Replace spaces with underscores
    label = re.sub(r'\s+', '_', label)
    return label

# Create a dictionary for mapping SAS Variable Names to cleaned Labels
mapping = {row['SAS Variable Name']: clean_label(row['Label']) for _, row in cols_df.iterrows()}

# Print the mapping dictionary to verify the changes
#print("Column Renaming Mapping:")
#for k, v in mapping.items():
#    print(f"{k}: {v}")
# Rename the columns in the actual data DataFrame
df.rename(columns=mapping, inplace=True)
df.head()


# ## **Heart Disease related features**<a id='Heart_Disease_related_features'></a>
# [Contents](#Contents)
# 
# After several days of research and analysis of the dataset's features, we have identified the following key features for heart disease assessment:
# 
# * **Target Variable (Dependent Variable):**
#     * Heart_disease: "Ever_Diagnosed_with_Angina_or_Coronary_Heart_Disease"
# * **Demographics:**
#     * Gender: Are_you_male_or_female
#     * Race: Computed_race_groups_used_for_internet_prevalence_tables
#     * Age: Imputed_Age_value_collapsed_above_80
# * **Medical History:**
#     * General_Health
#     * Have_Personal_Health_Care_Provider
#     * Could_Not_Afford_To_See_Doctor
#     * Length_of_time_since_last_routine_checkup
#     * Ever_Diagnosed_with_Heart_Attack
#     * Ever_Diagnosed_with_a_Stroke
#     * Ever_told_you_had_a_depressive_disorder
#     * Ever_told_you_have_kidney_disease
#     * Ever_told_you_had_diabetes
#     * Reported_Weight_in_Pounds
#     * Reported_Height_in_Feet_and_Inches
#     * Computed_body_mass_index_categories
#     * Difficulty_Walking_or_Climbing_Stairs
#     * Computed_Physical_Health_Status
#     * Computed_Mental_Health_Status
#     * Computed_Asthma_Status
# * **Life Style:**
#     * Leisure_Time_Physical_Activity_Calculated_Variable
#     * Smoked_at_Least_100_Cigarettes
#     * Computed_Smoking_Status
#     * Binge_Drinking_Calculated_Variable
#     * Computed_number_of_drinks_of_alcohol_beverages_per_week
#     * Exercise_in_Past_30_Days
#     * How_Much_Time_Do_You_Sleep
# 

# ## **Selection Heart disease related features**<a id='Selection_Heart_disease_related_features'></a>
# [Contents](#Contents)

# In[9]:


#Here, let's seelect the main features directly related to heart disease:
df = df[["Ever_Diagnosed_with_Angina_or_Coronary_Heart_Disease", # Target Variable
         "Are_you_male_or_female", #Demographics
         "Computed_race_groups_used_for_internet_prevalence_tables",#Demographics
         "Imputed_Age_value_collapsed_above_80",#Demographics
         "General_Health", #Medical History
         "Have_Personal_Health_Care_Provider",#Medical History
         "Could_Not_Afford_To_See_Doctor",#Medical History
         "Length_of_time_since_last_routine_checkup",#Medical History
         "Ever_Diagnosed_with_Heart_Attack",#Medical History
         "Ever_Diagnosed_with_a_Stroke",#Medical History
         "Ever_told_you_had_a_depressive_disorder",#Medical History
         "Ever_told_you_have_kidney_disease",#Medical History
         "Ever_told_you_had_diabetes",#Medical History
         "Reported_Weight_in_Pounds",#Medical History
         "Reported_Height_in_Feet_and_Inches",#Medical History
         "Computed_body_mass_index_categories",#Medical History
         "Difficulty_Walking_or_Climbing_Stairs",#Medical History
         "Computed_Physical_Health_Status",#Medical History
         "Computed_Mental_Health_Status",#Medical History
         "Computed_Asthma_Status",#Medical History
         "Leisure_Time_Physical_Activity_Calculated_Variable",#Life Style
         "Smoked_at_Least_100_Cigarettes",#Life Style
         "Computed_Smoking_Status",#Life Style
         "Binge_Drinking_Calculated_Variable",#Life Style
         "Computed_number_of_drinks_of_alcohol_beverages_per_week",#Life Style
         "Exercise_in_Past_30_Days",#Life Style
         "How_Much_Time_Do_You_Sleep"#Life Style
        ]]
df.head()


# In[10]:


#now, let's look at the shape of df after features selection:
shape = df.shape
print("Number of rows:", shape[0], "\nNumber of columns:", shape[1])


# ## **Imputing Missing Data, Transforming Columns and Features Engineering**<a id='Imputing_missing_Data_and_transforming_columns'></a>
# [Contents](#Contents)
# 
# In this step, we address missing data, map categorical values, and rename columns for improved data quality and clarity. The key actions taken are as follows:
# 
# * Replace Specific Values with NaN: Identify and replace erroneous or placeholder values with NaN to standardize missing data representation.
# * Calculate Value Distribution: Determine the distribution of existing values to understand the data's baseline state.
# * Impute Missing Values: Use a function to impute missing values based on the calculated distribution, ensuring the data remains representative of its original characteristics.
# * Map Categorical Values: Apply a mapping dictionary to convert numeric codes into meaningful categorical labels.
# * Rename Columns: Update column names to reflect their contents accurately and improve dataset readability.
# * Feature Engineering: Created new features that may enhance the predictive power of our models. These steps are essential for building a reliable and robust model for heart disease prediction.
# 

# ### **Distribution-Based Imputation**<a id='Distribution_Based_Imputation'></a>
# [Contents](#Contents)
# To deal with missing data in this project, we'll be using **Distribution-Based Imputation**:
# 
# * **Introduction**
#     * Distribution-Based: The imputation process relies on the existing distribution of the categories in the dataset.
#     * Imputation: The act of filling in missing values.
# * **Why This Method Works**
#     * Preserves Original Distribution: By using the observed proportions to guide the imputation, the method maintains the original distribution of gender categories.
#     * Random Imputation: Randomly selecting values based on the existing distribution prevents systematic biases that could arise from deterministic imputation methods.
#     * Scalability: This approach can be easily scaled to larger datasets and applied to other categorical variables with missing values.
# * **Advantages**
#     * Bias Minimization: Ensures that the imputed values do not skew the dataset in favor of any particular category.
#     * Simplicity: The method is straightforward to implement and understand.
#     * Flexibility: Can be adapted to any categorical variable with missing values.
# 
# This method is particularly useful in scenarios where preserving the natural distribution of data is crucial for subsequent analysis or modeling tasks. 

# In[11]:


#let's run below to examin each features again missing data count & percentage, unique count, data types:
summarize_df(df)


# ### **Column 1: Are_you_male_or_female**<a id='Column_1_Are_you_male_or_female'></a>
# [Contents](#Contents)
# 
# We have 4 versions of the same column, so now let's keep the least columns with missing data `21.58`

# In[12]:


# let's get the column names in a list:
print(df.columns)


# In[13]:


#let's select the main features related to heart disease:
df.columns = ['Ever_Diagnosed_with_Angina_or_Coronary_Heart_Disease', # This is my target variable!!!
       'Are_you_male_or_female_1', 'Are_you_male_or_female_2',
       'Are_you_male_or_female_3', 'Are_you_male_or_female_4',
       'Computed_race_groups_used_for_internet_prevalence_tables',
       'Imputed_Age_value_collapsed_above_80', 'General_Health',
       'Have_Personal_Health_Care_Provider', 'Could_Not_Afford_To_See_Doctor',
       'Length_of_time_since_last_routine_checkup',
       'Ever_Diagnosed_with_Heart_Attack', 'Ever_Diagnosed_with_a_Stroke',
       'Ever_told_you_had_a_depressive_disorder',
       'Ever_told_you_have_kidney_disease', 'Ever_told_you_had_diabetes',
       'Reported_Weight_in_Pounds', 'Reported_Height_in_Feet_and_Inches',
       'Computed_body_mass_index_categories',
       'Difficulty_Walking_or_Climbing_Stairs',
       'Computed_Physical_Health_Status', 'Computed_Mental_Health_Status',
       'Computed_Asthma_Status',
       'Leisure_Time_Physical_Activity_Calculated_Variable',
       'Smoked_at_Least_100_Cigarettes', 'Computed_Smoking_Status',
       'Binge_Drinking_Calculated_Variable',
       'Computed_number_of_drinks_of_alcohol_beverages_per_week',
       'Exercise_in_Past_30_Days', 'How_Much_Time_Do_You_Sleep']
#let's run below to examin each features again missing data count & percentage, unique count, data types:
summarize_df(df)


# Alright, as we can see above, now, let's drop 'Are_you_male_or_female_1', 'Are_you_male_or_female_2' and 'Are_you_male_or_female_4'

# In[14]:


#Let's drop the unnecessary columns:
columns_to_drop = ['Are_you_male_or_female_1', 'Are_you_male_or_female_2', 'Are_you_male_or_female_4']
df = df.drop(columns=columns_to_drop)

#let's run below to examin each features again missing data count & percentage, unique count, data types:
summarize_df(df)


# In[15]:


# view columns count:
df.Are_you_male_or_female_3.value_counts(dropna=False)


# **Are_you_male_or_female_3:**
# * 2: Femal
# * 1: Male
# * 3: Nonbinary
# * 7: Don’t know/Not Sure
# * 9: Refused
# 
# So based on above, let's change 7 and 9 to nan

# In[16]:


# Replace 7 and 9 with NaN
df['Are_you_male_or_female_3'].replace([7, 9], np.nan, inplace=True)
df.Are_you_male_or_female_3.value_counts(dropna=False)


# In[17]:


# Calculate the distribution of existing values
value_counts = df['Are_you_male_or_female_3'].value_counts(normalize=True, dropna=True)
print("Original distribution:\n", value_counts)


# In[18]:


# Function to impute missing values based on distribution
def impute_missing_gender(row):
    if pd.isna(row['Are_you_male_or_female_3']):
        return np.random.choice(value_counts.index, p=value_counts.values)
    else:
        return row['Are_you_male_or_female_3']

# Apply the imputation function
df['Are_you_male_or_female_3'] = df.apply(impute_missing_gender, axis=1)


# In[19]:


# Verify the imputation
imputed_value_counts = df['Are_you_male_or_female_3'].value_counts(dropna=False) # normalize=True
print("Distribution after imputation:\n", imputed_value_counts)


# Alright, as we can see above, no missing data on this column and the proportions reserved (Random imputation worked as expected).

# In[20]:


# Create a mapping dictionary:
gender_mapping = {2: 'female', 1: 'male', 3: 'nonbinary'}

# Apply the mapping to the gender column:
df['Are_you_male_or_female_3'] = df['Are_you_male_or_female_3'].map(gender_mapping)

# Rename the column:
df.rename(columns={'Are_you_male_or_female_3': 'gender'}, inplace=True)

df.head()


# In[21]:


#let's run below to examin each features again missing data count & percentage, unique count, data types:
summarize_df(df)


# ### **Column 2: Ever_Diagnosed_with_Angina_or_Coronary_Heart_Disease**<a id='Column_2_Ever_Diagnosed_with_Angina_or_Coronary_Heart_Disease'></a>
# [Contents](#Contents)

# In[22]:


#view column counts:
df.Ever_Diagnosed_with_Angina_or_Coronary_Heart_Disease.value_counts(dropna=False)


# **Ever_Diagnosed_with_Angina_or_Coronary_Heart_Disease:**
# * 2: No
# * 1: Yes
# * 7: Don’t know/Not Sure
# * 9: Refused
# 
# Alright, so next let's change 7 and 9 to nan:

# In[23]:


# Replace 7 and 9 with NaN
df['Ever_Diagnosed_with_Angina_or_Coronary_Heart_Disease'].replace([7, 9], np.nan, inplace=True)
df.Ever_Diagnosed_with_Angina_or_Coronary_Heart_Disease.value_counts(dropna=False)


# Alright, again, let's use  **Distribution-Based Imputation** for the above missing data:

# In[24]:


# Calculate the distribution of existing values
value_counts = df['Ever_Diagnosed_with_Angina_or_Coronary_Heart_Disease'].value_counts(normalize=True, dropna=True)
print("Original distribution:\n", value_counts)


# In[25]:


# Function to impute missing values based on distribution
def impute_missing(row):
    if pd.isna(row['Ever_Diagnosed_with_Angina_or_Coronary_Heart_Disease']):
        return np.random.choice(value_counts.index, p=value_counts.values)
    else:
        return row['Ever_Diagnosed_with_Angina_or_Coronary_Heart_Disease']


# In[26]:


# Apply the imputation function
df['Ever_Diagnosed_with_Angina_or_Coronary_Heart_Disease'] = df.apply(impute_missing, axis=1)


# In[27]:


# Verify the imputation
imputed_value_counts = df['Ever_Diagnosed_with_Angina_or_Coronary_Heart_Disease'].value_counts(dropna=False) # normalize=True
print("Distribution after imputation:\n", imputed_value_counts)


# In[28]:


# Verify the imputation
imputed_value_counts = df['Ever_Diagnosed_with_Angina_or_Coronary_Heart_Disease'].value_counts(dropna=False, normalize=True) # 
print("Distribution after imputation:\n", imputed_value_counts)


# In[29]:


# Create a mapping dictionary:
heart_disease_mapping = {2: 'no', 1: 'yes'}

# Apply the mapping to the "Ever_Diagnosed_with_Angina_or_Coronary_Heart_Disease" column:
df['Ever_Diagnosed_with_Angina_or_Coronary_Heart_Disease'] = df['Ever_Diagnosed_with_Angina_or_Coronary_Heart_Disease'].map(heart_disease_mapping)

# Rename the column:
df.rename(columns={'Ever_Diagnosed_with_Angina_or_Coronary_Heart_Disease': 'heart_disease'}, inplace=True)


# In[30]:


#let's run below to examin each features again missing data count & percentage, unique count, data types:
summarize_df(df)


# ### **Column 3: Computed_race_groups_used_for_internet_prevalence_tables**<a id='Column_3_Computed_race_groups_used_for_internet_prevalence_tables'></a>
# [Contents](#Contents)

# In[31]:


#view column counts:
df.Computed_race_groups_used_for_internet_prevalence_tables.value_counts(dropna=False)


# Alright, so good news is there's no missing data in this column

# **Computed_race_groups_used_for_internet_prevalence_tables:**
# * 1: white_only_non_hispanic
# * 2: black_only_non_hispanic
# * 3: american_indian_or_alaskan_native_only_non_hispanic
# * 4: asian_only_non_hispanic
# * 5: native_hawaiian_or_other_pacific_islander_only_non_hispanic
# * 6: multiracial_non_hispanic
# * 7: hispanic
# 

# In[32]:


# Create a mapping dictionary:
race_mapping = {1: 'white_only_non_hispanic',
2: 'black_only_non_hispanic',
3: 'american_indian_or_alaskan_native_only_non_hispanic',
4: 'asian_only_non_hispanic',
5: 'native_hawaiian_or_other_pacific_islander_only_non_hispanic',
6: 'multiracial_non_hispanic',
7: 'hispanic'}

# Apply the mapping to the race column:
df['Computed_race_groups_used_for_internet_prevalence_tables'] = df['Computed_race_groups_used_for_internet_prevalence_tables'].map(race_mapping)

# Rename the column:
df.rename(columns={'Computed_race_groups_used_for_internet_prevalence_tables': 'race'}, inplace=True)


# In[33]:


#view column counts:
df.race.value_counts(dropna=False)


# ### **column 4: Imputed_Age_value_collapsed_above_80**<a id='Column_4_Imputed_Age_value_collapsed_above_80'></a>
# [Contents](#Contents)

# In[34]:


#view column counts:
df.Imputed_Age_value_collapsed_above_80.value_counts(dropna=False)


# In[35]:


# Define bins and labels:
bins = [17, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 99]
labels = [
    'Age_18_to_24', 'Age_25_to_29', 'Age_30_to_34', 'Age_35_to_39',
    'Age_40_to_44', 'Age_45_to_49', 'Age_50_to_54', 'Age_55_to_59',
    'Age_60_to_64', 'Age_65_to_69', 'Age_70_to_74', 'Age_75_to_79',
    'Age_80_or_older'
         ]


# In[36]:


# Categorize the age values into bins:
df['age_category'] = pd.cut(df['Imputed_Age_value_collapsed_above_80'], bins=bins, labels=labels, right=True)
df.age_category.value_counts(dropna=False)


# ### **Column 5: General_Health**<a id='Column_5_General_Health'></a>
# [Contents](#Contents)

# In[37]:


#view column counts:
df.General_Health.value_counts(dropna=False)


# **General_Health:**
# * 1: excellent
# * 2: very_good
# * 3: good
# * 4: fair
# * 5: poor
# * 7: dont_know
# * 9: refused
# 
# so for 7, 9 let's convert to nan:
# 

# In[38]:


# Replace 7 and 9 with NaN
df['General_Health'].replace([7, 9], np.nan, inplace=True)
df.General_Health.value_counts(dropna=False)


# In[39]:


# Calculate the distribution of existing values
value_counts = df['General_Health'].value_counts(normalize=True, dropna=True)
print("Original General_Health:\n", value_counts)


# In[40]:


# Function to impute missing values based on distribution
def impute_missing(row):
    if pd.isna(row['General_Health']):
        return np.random.choice(value_counts.index, p=value_counts.values)
    else:
        return row['General_Health']


# In[41]:


# Apply the imputation function
df['General_Health'] = df.apply(impute_missing, axis=1)


# In[42]:


# Verify the imputation
imputed_value_counts = df['General_Health'].value_counts(dropna=False) # normalize=True
print("Distribution after imputation:\n", imputed_value_counts)


# In[43]:


# Create a mapping dictionary:
health_mapping = {1: 'excellent',
                  2: 'very_good',
                  3: 'good',
                  4: 'fair',
                  5: 'poor'
                 }


# Apply the mapping to the health column:
df['General_Health'] = df['General_Health'].map(health_mapping)

# Rename the column:
df.rename(columns={'General_Health': 'general_health'}, inplace=True)


# In[44]:


#view column counts:
df.general_health.value_counts(dropna=False)


# ### **Column 6: Have_Personal_Health_Care_Provider**<a id='Column_6_Have_Personal_Health_Care_Provider'></a>	
# [Contents](#Contents)

# In[45]:


#view column counts:
df.Have_Personal_Health_Care_Provider.value_counts(dropna=False)


# **Have_Personal_Health_Care_Provider:**
# * 1: yes_only_one
# * 2: more_than_one
# * 3: no
# * 7: dont_know
# * 9: refused
# 
# so for 7, 9 let's convert to nan:

# In[46]:


# Replace 7 and 9 with NaN
df['Have_Personal_Health_Care_Provider'].replace([7, 9], np.nan, inplace=True)
df.Have_Personal_Health_Care_Provider.value_counts(dropna=False)


# In[47]:


# Calculate the distribution of existing values
value_counts = df['Have_Personal_Health_Care_Provider'].value_counts(normalize=True, dropna=True)
print("Original Have_Personal_Health_Care_Provider:\n", value_counts)


# In[48]:


# Function to impute missing values based on distribution
def impute_missing(row):
    if pd.isna(row['Have_Personal_Health_Care_Provider']):
        return np.random.choice(value_counts.index, p=value_counts.values)
    else:
        return row['Have_Personal_Health_Care_Provider']


# In[49]:


# Apply the imputation function
df['Have_Personal_Health_Care_Provider'] = df.apply(impute_missing, axis=1)


# In[50]:


# Verify the imputation
imputed_value_counts = df['Have_Personal_Health_Care_Provider'].value_counts(dropna=False) # normalize=True
print("Distribution after imputation:\n", imputed_value_counts)


# In[51]:


# Verify the imputation
imputed_value_counts = df['Have_Personal_Health_Care_Provider'].value_counts(dropna=False,normalize=True) # 
print("Distribution after imputation:\n", imputed_value_counts)


# In[52]:


# Create a mapping dictionary:
porvider_mapping = {1: 'yes_only_one',
                  2: 'more_than_one',
                  3: 'no'
                 }

# Apply the mapping to the provider column:
df['Have_Personal_Health_Care_Provider'] = df['Have_Personal_Health_Care_Provider'].map(porvider_mapping)

# Rename the column:
df.rename(columns={'Have_Personal_Health_Care_Provider': 'health_care_provider'}, inplace=True)


# ### **Column 7: Could_Not_Afford_To_See_Doctor**<a id='Column_7_Could_Not_Afford_To_See_Doctor'></a>
# [Contents](#Contents)

# In[53]:


#view column counts:
df.Could_Not_Afford_To_See_Doctor.value_counts(dropna=False)


# **Could_Not_Afford_To_See_Doctor:**
# * 1: yes
# * 2: no
# * 7: dont_know
# * 9: refused
# 
# so for 7, 9 let's convert to nan:

# In[54]:


# Replace 7 and 9 with NaN
df['Could_Not_Afford_To_See_Doctor'].replace([7, 9], np.nan, inplace=True)
df.Could_Not_Afford_To_See_Doctor.value_counts(dropna=False)


# In[55]:


# Calculate the distribution of existing values
value_counts = df['Could_Not_Afford_To_See_Doctor'].value_counts(normalize=True, dropna=True)
print("Original Could_Not_Afford_To_See_Doctor:\n", value_counts)


# In[56]:


# Function to impute missing values based on distribution
def impute_missing(row):
    if pd.isna(row['Could_Not_Afford_To_See_Doctor']):
        return np.random.choice(value_counts.index, p=value_counts.values)
    else:
        return row['Could_Not_Afford_To_See_Doctor']


# In[57]:


# Apply the imputation function
df['Could_Not_Afford_To_See_Doctor'] = df.apply(impute_missing, axis=1)


# In[58]:


# Verify the imputation
imputed_value_counts = df['Could_Not_Afford_To_See_Doctor'].value_counts(dropna=False) # normalize=True
print("Distribution after imputation:\n", imputed_value_counts)


# In[59]:


# Verify the imputation
imputed_value_counts = df['Could_Not_Afford_To_See_Doctor'].value_counts(dropna=False,normalize=True) # 
print("Distribution after imputation:\n", imputed_value_counts)


# In[60]:


# Create a mapping dictionary:
doctor_mapping = {1: 'yes',
                  2: 'no'
                 }

# Apply the mapping to the doctor column:
df['Could_Not_Afford_To_See_Doctor'] = df['Could_Not_Afford_To_See_Doctor'].map(doctor_mapping)

# Rename the column:
df.rename(columns={'Could_Not_Afford_To_See_Doctor': 'could_not_afford_to_see_doctor'}, inplace=True)


# ### **Column 8: Length_of_time_since_last_routine_checkup**<a id='Column_8_Length_of_time_since_last_routine_checkup'></a>
# [Contents](#Contents)

# In[61]:


#view column counts:
df.Length_of_time_since_last_routine_checkup.value_counts(dropna=False)


# **Could_Not_Afford_To_See_Doctor:**
# * 1: 'past_year',
# * 2: 'past_2_years',
# * 3: 'past_5_years',
# * 4: '5+_years_ago',
# * 7: 'dont_know',
# * 8: 'never',
# * 9: 'refused',
# so for 7, 9 let's convert to nan:

# In[62]:


#Replace 7 and 9 with NaN:
df['Length_of_time_since_last_routine_checkup'].replace([7, 9], np.nan, inplace=True)
df.Length_of_time_since_last_routine_checkup.value_counts(dropna=False)


# In[63]:


# Calculate the distribution of existing values:
value_counts = df['Length_of_time_since_last_routine_checkup'].value_counts(normalize=True, dropna=True)
print("Original Length_of_time_since_last_routine_checkup:\n", value_counts)


# In[64]:


# Function to impute missing values based on distribution:
def impute_missing(row):
    if pd.isna(row['Length_of_time_since_last_routine_checkup']):
        return np.random.choice(value_counts.index, p=value_counts.values)
    else:
        return row['Length_of_time_since_last_routine_checkup']


# In[65]:


# Apply the imputation function:
df['Length_of_time_since_last_routine_checkup'] = df.apply(impute_missing, axis=1)


# In[66]:


# Verify the imputation:
imputed_value_counts = df['Length_of_time_since_last_routine_checkup'].value_counts(dropna=False) # normalize=True
print("Distribution after imputation:\n", imputed_value_counts)


# In[67]:


# Verify the imputation:
imputed_value_counts = df['Length_of_time_since_last_routine_checkup'].value_counts(dropna=False,normalize=True) # 
print("Distribution after imputation:\n", imputed_value_counts)


# In[68]:


# Create a mapping dictionary:
checkup_mapping = {1: 'past_year',
                   2: 'past_2_years',
                   3: 'past_5_years',
                   4: '5+_years_ago',
                   8: 'never',
                 }

# Apply the mapping to the checkup_mapping column:
df['Length_of_time_since_last_routine_checkup'] = df['Length_of_time_since_last_routine_checkup'].map(checkup_mapping)

# Rename the column:
df.rename(columns={'Length_of_time_since_last_routine_checkup': 'length_of_time_since_last_routine_checkup'}, inplace=True)


# In[69]:


#view column counts:
df['length_of_time_since_last_routine_checkup'].value_counts(dropna=False,normalize=True)


# ### **Column 9: Ever_Diagnosed_with_Heart_Attack**<a id='Column_9_Ever_Diagnosed_with_Heart_Attack'></a>
# [Contents](#Contents)

# In[70]:


#view column counts:
df['Ever_Diagnosed_with_Heart_Attack'].value_counts(dropna=False)


# **Ever_Diagnosed_with_Heart_Attack:**
# * 1: yes
# * 2: no
# * 7: dont_know
# * 9: refused
# 
# so for 7, 9 let's convert to nan:

# In[71]:


#Replace 7 and 9 with NaN:
df['Ever_Diagnosed_with_Heart_Attack'].replace([7, 9], np.nan, inplace=True)
df.Ever_Diagnosed_with_Heart_Attack.value_counts(dropna=False)


# In[72]:


# Calculate the distribution of existing values:
value_counts = df['Ever_Diagnosed_with_Heart_Attack'].value_counts(normalize=True, dropna=True)
print("Original Length_of_time_since_last_routine_checkup:\n", value_counts)


# In[73]:


# Function to impute missing values based on distribution:
def impute_missing(row):
    if pd.isna(row['Ever_Diagnosed_with_Heart_Attack']):
        return np.random.choice(value_counts.index, p=value_counts.values)
    else:
        return row['Ever_Diagnosed_with_Heart_Attack']


# In[74]:


# Apply the imputation function:
df['Ever_Diagnosed_with_Heart_Attack'] = df.apply(impute_missing, axis=1)


# In[75]:


# Verify the imputation:
imputed_value_counts = df['Ever_Diagnosed_with_Heart_Attack'].value_counts(dropna=False) # normalize=True
print("Distribution after imputation:\n", imputed_value_counts)


# In[76]:


# Verify the imputation:
imputed_value_counts = df['Ever_Diagnosed_with_Heart_Attack'].value_counts(dropna=False,normalize=True) # 
print("Distribution after imputation:\n", imputed_value_counts)


# In[77]:


# Create a mapping dictionary:
heart_attack_mapping = {1: 'yes',
                   2: 'no',

                 }

# Apply the mapping to the heart_attack_mapping column:
df['Ever_Diagnosed_with_Heart_Attack'] = df['Ever_Diagnosed_with_Heart_Attack'].map(heart_attack_mapping)

# Rename the column:
df.rename(columns={'Ever_Diagnosed_with_Heart_Attack': 'ever_diagnosed_with_heart_attack'}, inplace=True)


# In[78]:


#view column counts:
df['ever_diagnosed_with_heart_attack'].value_counts(dropna=False,normalize=True) # 


# ### **Column 10: Ever_Diagnosed_with_a_Stroke**<a id='Column_10_Ever_Diagnosed_with_a_Stroke'></a>
# [Contents](#Contents)

# In[79]:


#view column counts:
df['Ever_Diagnosed_with_a_Stroke'].value_counts(dropna=False)


# **Ever_Diagnosed_with_Heart_Attack:**
# * 1: yes
# * 2: no
# * 7: dont_know
# * 9: refused
# 
# so for 7, 9 let's convert to nan:

# In[80]:


#Replace 7 and 9 with NaN:
df['Ever_Diagnosed_with_a_Stroke'].replace([7, 9], np.nan, inplace=True)
df.Ever_Diagnosed_with_a_Stroke.value_counts(dropna=False)


# In[81]:


# Calculate the distribution of existing values:
value_counts = df['Ever_Diagnosed_with_a_Stroke'].value_counts(normalize=True, dropna=True)
print("Original Ever_Diagnosed_with_a_Stroke:\n", value_counts)


# In[82]:


# Function to impute missing values based on distribution:
def impute_missing(row):
    if pd.isna(row['Ever_Diagnosed_with_a_Stroke']):
        return np.random.choice(value_counts.index, p=value_counts.values)
    else:
        return row['Ever_Diagnosed_with_a_Stroke']


# In[83]:


# Apply the imputation function:
df['Ever_Diagnosed_with_a_Stroke'] = df.apply(impute_missing, axis=1)


# In[84]:


# Verify the imputation:
imputed_value_counts = df['Ever_Diagnosed_with_a_Stroke'].value_counts(dropna=False) # normalize=True
print("Distribution after imputation:\n", imputed_value_counts)


# In[85]:


# Verify the imputation:
imputed_value_counts = df['Ever_Diagnosed_with_a_Stroke'].value_counts(dropna=False,normalize=True) # 
print("Distribution after imputation:\n", imputed_value_counts)


# In[86]:


# Create a mapping dictionary:
stroke_mapping = {1: 'yes',
                   2: 'no',

                 }

# Apply the mapping to the stroke column:
df['Ever_Diagnosed_with_a_Stroke'] = df['Ever_Diagnosed_with_a_Stroke'].map(stroke_mapping)

# Rename the column:
df.rename(columns={'Ever_Diagnosed_with_a_Stroke': 'ever_diagnosed_with_a_stroke'}, inplace=True)


# In[87]:


#view column counts:
df['ever_diagnosed_with_a_stroke'].value_counts(dropna=False,normalize=True) # 


# ### **Column 11: Ever_told_you_had_a_depressive_disorder**<a id='Column_11_Ever_told_you_had_a_depressive_disorder'></a>
# [Contents](#Contents)

# In[88]:


#view column counts:
value_counts_with_percentage(df, 'Ever_told_you_had_a_depressive_disorder')


# **Ever_told_you_had_a_depressive_disorder:**
# * 1: yes
# * 2: no
# * 7: dont_know
# * 9: refused
# 
# so for 7, 9 let's convert to nan:

# In[89]:


#Replace 7 and 9 with NaN:
df['Ever_told_you_had_a_depressive_disorder'].replace([7, 9], np.nan, inplace=True)
df.Ever_told_you_had_a_depressive_disorder.value_counts(dropna=False)


# In[90]:


# Calculate the distribution of existing values:
value_counts = df['Ever_told_you_had_a_depressive_disorder'].value_counts(normalize=True, dropna=True)
print("Original Ever_told_you_had_a_depressive_disorder:\n", value_counts)


# In[91]:


# Function to impute missing values based on distribution:
def impute_missing(row):
    if pd.isna(row['Ever_told_you_had_a_depressive_disorder']):
        return np.random.choice(value_counts.index, p=value_counts.values)
    else:
        return row['Ever_told_you_had_a_depressive_disorder']


# In[92]:


# Apply the imputation function:
df['Ever_told_you_had_a_depressive_disorder'] = df.apply(impute_missing, axis=1)


# In[93]:


# Verify the imputation:
imputed_value_counts = df['Ever_told_you_had_a_depressive_disorder'].value_counts(dropna=False) # normalize=True
print("Distribution after imputation:\n", imputed_value_counts)


# In[94]:


# Verify the imputation:
imputed_value_counts = df['Ever_told_you_had_a_depressive_disorder'].value_counts(dropna=False,normalize=True) # 
print("Distribution after imputation:\n", imputed_value_counts)


# In[95]:


# Create a mapping dictionary:
depressive_disorder_mapping = {1: 'yes',
                   2: 'no',

                 }

# Apply the mapping to the depressive_disorder column:
df['Ever_told_you_had_a_depressive_disorder'] = df['Ever_told_you_had_a_depressive_disorder'].map(depressive_disorder_mapping)

# Rename the column:
df.rename(columns={'Ever_told_you_had_a_depressive_disorder': 'ever_told_you_had_a_depressive_disorder'}, inplace=True)


# In[96]:


#view column counts & percentage:
value_counts_with_percentage(df, 'ever_told_you_had_a_depressive_disorder')


# ### **Column 12: Ever_told_you_have_kidney_disease**<a id='Column_12_Ever_told_you_have_kidney_disease'></a>
# [Contents](#Contents)

# In[97]:


#view column counts & percentage:
value_counts_with_percentage(df, 'Ever_told_you_have_kidney_disease')


# **Ever_told_you_had_a_depressive_disorder:**
# * 1: yes
# * 2: no
# * 7: dont_know
# * 9: refused
# 
# so for 7, 9 let's convert to nan:

# In[98]:


#Replace 7 and 9 with NaN:
df['Ever_told_you_have_kidney_disease'].replace([7, 9], np.nan, inplace=True)
df.Ever_told_you_have_kidney_disease.value_counts(dropna=False)


# In[99]:


# Calculate the distribution of existing values:
value_counts = df['Ever_told_you_have_kidney_disease'].value_counts(normalize=True, dropna=True)
print("Original Ever_told_you_have_kidney_disease:\n", value_counts)


# In[100]:


# Function to impute missing values based on distribution:
def impute_missing(row):
    if pd.isna(row['Ever_told_you_have_kidney_disease']):
        return np.random.choice(value_counts.index, p=value_counts.values)
    else:
        return row['Ever_told_you_have_kidney_disease']


# In[101]:


# Apply the imputation function:
df['Ever_told_you_have_kidney_disease'] = df.apply(impute_missing, axis=1)


# In[102]:


# Verify the imputation:
imputed_value_counts = df['Ever_told_you_have_kidney_disease'].value_counts(dropna=False) # normalize=True
print("Distribution after imputation:\n", imputed_value_counts)


# In[103]:


# Verify the imputation:
imputed_value_counts = df['Ever_told_you_have_kidney_disease'].value_counts(dropna=False,normalize=True) # 
print("Distribution after imputation:\n", imputed_value_counts)


# In[104]:


# Create a mapping dictionary:
kidney_mapping = {1: 'yes',
                   2: 'no',

                 }

# Apply the mapping to the kidney column:
df['Ever_told_you_have_kidney_disease'] = df['Ever_told_you_have_kidney_disease'].map(kidney_mapping)

# Rename the column:
df.rename(columns={'Ever_told_you_have_kidney_disease': 'ever_told_you_have_kidney_disease'}, inplace=True)


# In[105]:


#view column counts & percentage:
value_counts_with_percentage(df, 'ever_told_you_have_kidney_disease')


# ### **Column 13: Ever_told_you_had_diabetes**<a id='Column_13_Ever_told_you_had_diabetes'></a>
# [Contents](#Contents)

# In[106]:


#view column counts & percentage:
value_counts_with_percentage(df, 'Ever_told_you_had_diabetes')


# **Ever_told_you_had_diabetes:**
# * 1: 'yes',
# * 2: 'yes_during_pregnancy',
# * 3: 'no',
# * 4: 'no_prediabetes',
# * 7: 'dont_know',
# * 9: 'refused',
# 
# so for 7, 9 let's convert to nan:

# In[107]:


#Replace 7 and 9 with NaN:
df['Ever_told_you_had_diabetes'].replace([7, 9], np.nan, inplace=True)
df.Ever_told_you_had_diabetes.value_counts(dropna=False)


# In[108]:


# Calculate the distribution of existing values:
value_counts = df['Ever_told_you_had_diabetes'].value_counts(normalize=True, dropna=True)
print("Original Ever_told_you_have_kidney_disease:\n", value_counts)


# In[109]:


# Function to impute missing values based on distribution:
def impute_missing(row):
    if pd.isna(row['Ever_told_you_had_diabetes']):
        return np.random.choice(value_counts.index, p=value_counts.values)
    else:
        return row['Ever_told_you_had_diabetes']


# In[110]:


# Apply the imputation function:
df['Ever_told_you_had_diabetes'] = df.apply(impute_missing, axis=1)


# In[111]:


# Verify the imputation:
imputed_value_counts = df['Ever_told_you_had_diabetes'].value_counts(dropna=False) # normalize=True
print("Distribution after imputation:\n", imputed_value_counts)


# In[112]:


# Verify the imputation:
imputed_value_counts = df['Ever_told_you_had_diabetes'].value_counts(dropna=False,normalize=True) # 
print("Distribution after imputation:\n", imputed_value_counts)


# In[113]:


# Create a mapping dictionary:
diabetes_mapping = {1: 'yes',
                  2: 'yes_during_pregnancy',
                  3: 'no',
                  4: 'no_prediabetes',

                 }

# Apply the mapping to the diabetes column:
df['Ever_told_you_had_diabetes'] = df['Ever_told_you_had_diabetes'].map(diabetes_mapping)

# Rename the column:
df.rename(columns={'Ever_told_you_had_diabetes': 'ever_told_you_had_diabetes'}, inplace=True)


# In[114]:


#view column counts & percentage:
value_counts_with_percentage(df, 'ever_told_you_had_diabetes')


# ### **Column 14: Computed_body_mass_index_categories**<a id='Column_14_Computed_body_mass_index_categories'></a>
# [Contents](#Contents)

# In[115]:


#view column counts & percentage:
value_counts_with_percentage(df, 'Computed_body_mass_index_categories')


# **Computed_body_mass_index_categories:**
# * 1: 'underweight_bmi_less_than_18_5',
# * 2: 'normal_weight_bmi_18_5_to_24_9',
# * 3: 'overweight_bmi_25_to_29_9',
# * 4: 'obese_bmi_30_or_more',
# 

# In[116]:


# Calculate the distribution of existing values:
value_counts = df['Computed_body_mass_index_categories'].value_counts(normalize=True, dropna=True)
print("Original Computed_body_mass_index_categories:\n", value_counts)


# In[117]:


# Function to impute missing values based on distribution:
def impute_missing(row):
    if pd.isna(row['Computed_body_mass_index_categories']):
        return np.random.choice(value_counts.index, p=value_counts.values)
    else:
        return row['Computed_body_mass_index_categories']


# In[118]:


# Apply the imputation function:
df['Computed_body_mass_index_categories'] = df.apply(impute_missing, axis=1)


# In[119]:


# Verify the imputation:
imputed_value_counts = df['Computed_body_mass_index_categories'].value_counts(dropna=False) # normalize=True
print("Distribution after imputation:\n", imputed_value_counts)


# In[120]:


# Verify the imputation:
imputed_value_counts = df['Computed_body_mass_index_categories'].value_counts(dropna=False,normalize=True) # 
print("Distribution after imputation:\n", imputed_value_counts)


# In[121]:


# Create a mapping dictionary:
bmi_mapping = {1: 'underweight_bmi_less_than_18_5',
                    2: 'normal_weight_bmi_18_5_to_24_9',
                    3: 'overweight_bmi_25_to_29_9',
                    4: 'obese_bmi_30_or_more',

                 }

# Apply the mapping to the bmi column:
df['Computed_body_mass_index_categories'] = df['Computed_body_mass_index_categories'].map(bmi_mapping)

# Rename the column:
df.rename(columns={'Computed_body_mass_index_categories': 'BMI'}, inplace=True)


# In[122]:


#view column counts & percentage:
value_counts_with_percentage(df, 'BMI')


# ### **Column 15: Difficulty_Walking_or_Climbing_Stairs**<a id='Column_15_Difficulty_Walking_or_Climbing_Stairs'></a>
# [Contents](#Contents)

# In[123]:


#view column counts & percentage:
value_counts_with_percentage(df, 'Difficulty_Walking_or_Climbing_Stairs')


# **Difficulty_Walking_or_Climbing_Stairs:**
# * 1: yes
# * 2: no
# * 7: dont_know
# * 9: refused
# 
# so for 7, 9 let's convert to nan:

# In[124]:


#Replace 7 and 9 with NaN:
df['Difficulty_Walking_or_Climbing_Stairs'].replace([7, 9], np.nan, inplace=True)
df.Difficulty_Walking_or_Climbing_Stairs.value_counts(dropna=False)


# In[125]:


# Calculate the distribution of existing values:
value_counts = df['Difficulty_Walking_or_Climbing_Stairs'].value_counts(normalize=True, dropna=True)
print("Original Difficulty_Walking_or_Climbing_Stairs:\n", value_counts)


# In[126]:


# Function to impute missing values based on distribution:
def impute_missing(row):
    if pd.isna(row['Difficulty_Walking_or_Climbing_Stairs']):
        return np.random.choice(value_counts.index, p=value_counts.values)
    else:
        return row['Difficulty_Walking_or_Climbing_Stairs']


# In[127]:


# Apply the imputation function:
df['Difficulty_Walking_or_Climbing_Stairs'] = df.apply(impute_missing, axis=1)


# In[128]:


# Verify the imputation:
imputed_value_counts = df['Difficulty_Walking_or_Climbing_Stairs'].value_counts(dropna=False) # normalize=True
print("Distribution after imputation:\n", imputed_value_counts)


# In[129]:


# Verify the imputation:
imputed_value_counts = df['Difficulty_Walking_or_Climbing_Stairs'].value_counts(dropna=False,normalize=True) # 
print("Distribution after imputation:\n", imputed_value_counts)


# In[130]:


# Create a mapping dictionary:
climbing_mapping = {1: 'yes',
                   2: 'no',

                 }

# Apply the mapping to the climbing_mapping column:
df['Difficulty_Walking_or_Climbing_Stairs'] = df['Difficulty_Walking_or_Climbing_Stairs'].map(climbing_mapping)

# Rename the column:
df.rename(columns={'Difficulty_Walking_or_Climbing_Stairs': 'difficulty_walking_or_climbing_stairs'}, inplace=True)


# In[131]:


#view column counts & percentage:
value_counts_with_percentage(df, 'difficulty_walking_or_climbing_stairs')


# ### **Column 16: Computed_Physical_Health_Status**<a id='Column_16_Computed_Physical_Health_Status'></a>
# [Contents](#Contents)

# In[132]:


#view column counts & percentage:
value_counts_with_percentage(df, 'Computed_Physical_Health_Status')


# **Computed_Physical_Health_Status:**
# * 1: 'zero_days_not_good',
# * 2: '1_to_13_days_not_good',
# * 3: '14_plus_days_not_good',
# * 9: 'dont_know'
# 
# so for 9 let's convert to nan:

# In[133]:


#Replace 7 and 9 with NaN:
df['Computed_Physical_Health_Status'].replace([9], np.nan, inplace=True)
df.Computed_Physical_Health_Status.value_counts(dropna=False)


# In[134]:


# Calculate the distribution of existing values:
value_counts = df['Computed_Physical_Health_Status'].value_counts(normalize=True, dropna=True)
print("Original Computed_Physical_Health_Status:\n", value_counts)


# In[135]:


# Function to impute missing values based on distribution:
def impute_missing(row):
    if pd.isna(row['Computed_Physical_Health_Status']):
        return np.random.choice(value_counts.index, p=value_counts.values)
    else:
        return row['Computed_Physical_Health_Status']


# In[136]:


# Apply the imputation function:
df['Computed_Physical_Health_Status'] = df.apply(impute_missing, axis=1)


# In[137]:


# Verify the imputation:
imputed_value_counts = df['Computed_Physical_Health_Status'].value_counts(dropna=False) # normalize=True
print("Distribution after imputation:\n", imputed_value_counts)


# In[138]:


# Verify the imputation:
imputed_value_counts = df['Computed_Physical_Health_Status'].value_counts(dropna=False,normalize=True) # 
print("Distribution after imputation:\n", imputed_value_counts)


# In[139]:


# Create a mapping dictionary:
health_status_mapping = {1: 'zero_days_not_good',
                    2: '1_to_13_days_not_good',
                    3: '14_plus_days_not_good',

                 }

# Apply the mapping to the health_status_mapping column:
df['Computed_Physical_Health_Status'] = df['Computed_Physical_Health_Status'].map(health_status_mapping)

# Rename the column:
df.rename(columns={'Computed_Physical_Health_Status': 'physical_health_status'}, inplace=True)


# In[140]:


#view column counts & percentage:
value_counts_with_percentage(df, 'physical_health_status')


# ### **Column 17: Computed_Mental_Health_Status**<a id='Column_17_Computed_Mental_Health_Status'></a>	
# [Contents](#Contents)

# In[141]:


#view column counts & percentage:
value_counts_with_percentage(df, 'Computed_Mental_Health_Status')


# **Computed_Physical_Health_Status:**
# * 1: 'zero_days_not_good',
# * 2: '1_to_13_days_not_good',
# * 3: '14_plus_days_not_good',
# * 9: 'dont_know'
# 
# so for 9 let's convert to nan:

# In[142]:


#Replace 7 and 9 with NaN:
df['Computed_Mental_Health_Status'].replace([9], np.nan, inplace=True)
df.Computed_Mental_Health_Status.value_counts(dropna=False)


# In[143]:


# Calculate the distribution of existing values:
value_counts = df['Computed_Mental_Health_Status'].value_counts(normalize=True, dropna=True)
print("Original Computed_Mental_Health_Status:\n", value_counts)


# In[144]:


# Function to impute missing values based on distribution:
def impute_missing(row):
    if pd.isna(row['Computed_Mental_Health_Status']):
        return np.random.choice(value_counts.index, p=value_counts.values)
    else:
        return row['Computed_Mental_Health_Status']


# In[145]:


# Apply the imputation function:
df['Computed_Mental_Health_Status'] = df.apply(impute_missing, axis=1)


# In[146]:


# Verify the imputation:
imputed_value_counts = df['Computed_Mental_Health_Status'].value_counts(dropna=False) # normalize=True
print("Distribution after imputation:\n", imputed_value_counts)


# In[147]:


# Verify the imputation:
imputed_value_counts = df['Computed_Mental_Health_Status'].value_counts(dropna=False,normalize=True) # 
print("Distribution after imputation:\n", imputed_value_counts)


# In[148]:


# Create a mapping dictionary:
m_health_status_mapping = {1: 'zero_days_not_good',
                    2: '1_to_13_days_not_good',
                    3: '14_plus_days_not_good',

                 }

# Apply the mapping to the m_health_status_mapping column:
df['Computed_Mental_Health_Status'] = df['Computed_Mental_Health_Status'].map(m_health_status_mapping)

# Rename the column:
df.rename(columns={'Computed_Mental_Health_Status': 'mental_health_status'}, inplace=True)


# In[149]:


#view column counts & percentage:
value_counts_with_percentage(df, 'mental_health_status')


# ### **Column 18: Computed_Asthma_Status**<a id='Column_18_Computed_Asthma_Status'></a>
# [Contents](#Contents)

# In[150]:


#view column counts & percentage:
value_counts_with_percentage(df, 'Computed_Asthma_Status')


# **Computed_Asthma_Status:**
# * 1: 'current_asthma',
# * 2: 'former_asthma',
# * 3: 'never_asthma',
# * 9: 'dont_know_refused_missing'
# 
# so for 9 let's convert to nan:

# In[151]:


#Replace 7 and 9 with NaN:
df['Computed_Asthma_Status'].replace([9], np.nan, inplace=True)
df.Computed_Asthma_Status.value_counts(dropna=False)


# In[152]:


# Calculate the distribution of existing values:
value_counts = df['Computed_Asthma_Status'].value_counts(normalize=True, dropna=True)
print("Original Computed_Asthma_Status:\n", value_counts)


# In[153]:


# Function to impute missing values based on distribution:
def impute_missing(row):
    if pd.isna(row['Computed_Asthma_Status']):
        return np.random.choice(value_counts.index, p=value_counts.values)
    else:
        return row['Computed_Asthma_Status']


# In[154]:


# Apply the imputation function:
df['Computed_Asthma_Status'] = df.apply(impute_missing, axis=1)


# In[155]:


# Verify the imputation:
imputed_value_counts = df['Computed_Asthma_Status'].value_counts(dropna=False) # normalize=True
print("Distribution after imputation:\n", imputed_value_counts)


# In[156]:


# Verify the imputation:
imputed_value_counts = df['Computed_Asthma_Status'].value_counts(dropna=False,normalize=True) # 
print("Distribution after imputation:\n", imputed_value_counts)


# In[157]:


# Create a mapping dictionary:
Asthma_Status_mapping = {1: 'current_asthma',
                           2: 'former_asthma',
                           3: 'never_asthma',

                 }

# Apply the mapping to the Asthma_Status_mapping column:
df['Computed_Asthma_Status'] = df['Computed_Asthma_Status'].map(Asthma_Status_mapping)

# Rename the column:
df.rename(columns={'Computed_Asthma_Status': 'asthma_Status'}, inplace=True)


# In[158]:


#view column counts & percentage:
value_counts_with_percentage(df, 'asthma_Status')


# ### **Column 19: Exercise_in_Past_30_Days**<a id='Column_19_Exercise_in_Past_30_Days'></a>
# [Contents](#Contents)

# In[159]:


#view column counts & percentage:
value_counts_with_percentage(df, 'Exercise_in_Past_30_Days')


# **Exercise_in_Past_30_Days:**
# * 1: 'yes',
# * 2: 'no',
# * 7: 'dont_know'
# * 9: 'refused_missing'
# 
# so for 7, 9 let's convert to nan:

# In[160]:


#Replace 7 and 9 with NaN:
df['Exercise_in_Past_30_Days'].replace([7, 9], np.nan, inplace=True)
df.Exercise_in_Past_30_Days.value_counts(dropna=False)


# In[161]:


# Calculate the distribution of existing values:
value_counts = df['Exercise_in_Past_30_Days'].value_counts(normalize=True, dropna=True)
print("Original Exercise_in_Past_30_Days:\n", value_counts)


# In[162]:


# Function to impute missing values based on distribution:
def impute_missing(row):
    if pd.isna(row['Exercise_in_Past_30_Days']):
        return np.random.choice(value_counts.index, p=value_counts.values)
    else:
        return row['Exercise_in_Past_30_Days']


# In[163]:


# Apply the imputation function:
df['Exercise_in_Past_30_Days'] = df.apply(impute_missing, axis=1)


# In[164]:


# Verify the imputation:
imputed_value_counts = df['Exercise_in_Past_30_Days'].value_counts(dropna=False) # normalize=True
print("Distribution after imputation:\n", imputed_value_counts)


# In[165]:


# Verify the imputation:
imputed_value_counts = df['Exercise_in_Past_30_Days'].value_counts(dropna=False,normalize=True) # 
print("Distribution after imputation:\n", imputed_value_counts)


# In[166]:


# Create a mapping dictionary:
exercise_Status_mapping = {1: 'yes',
                           2: 'no',

                 }

# Apply the mapping to the exercise_Status_mapping column:
df['Exercise_in_Past_30_Days'] = df['Exercise_in_Past_30_Days'].map(exercise_Status_mapping)

# Rename the column:
df.rename(columns={'Exercise_in_Past_30_Days': 'exercise_status_in_past_30_Days'}, inplace=True)


# In[167]:


#view column counts & percentage:
value_counts_with_percentage(df, 'exercise_status_in_past_30_Days')


# ### **Column 20: Computed_Smoking_Status**<a id='Column_20_Computed_Smoking_Status'></a>
# [Contents](#Contents)

# In[168]:


#view column counts & percentage:
value_counts_with_percentage(df, 'Computed_Smoking_Status')


# **Computed_Smoking_Status:**
# * 1: 'current_smoker_every_day',
# * 2: 'current_smoker_some_days',
# * 3: 'former_smoker',
# * 4: 'never_smoked',
# * 9: 'dont_know_refused_missing'
# 
# so for 9 let's convert to nan:

# In[169]:


#Replace 7 and 9 with NaN:
df['Computed_Smoking_Status'].replace([9], np.nan, inplace=True)
df.Computed_Smoking_Status.value_counts(dropna=False)


# In[170]:


# Calculate the distribution of existing values:
value_counts = df['Computed_Smoking_Status'].value_counts(normalize=True, dropna=True)
print("Original Computed_Smoking_Status:\n", value_counts)


# In[171]:


# Function to impute missing values based on distribution:
def impute_missing(row):
    if pd.isna(row['Computed_Smoking_Status']):
        return np.random.choice(value_counts.index, p=value_counts.values)
    else:
        return row['Computed_Smoking_Status']


# In[172]:


# Apply the imputation function:
df['Computed_Smoking_Status'] = df.apply(impute_missing, axis=1)


# In[173]:


# Verify the imputation:
imputed_value_counts = df['Computed_Smoking_Status'].value_counts(dropna=False) # normalize=True
print("Distribution after imputation:\n", imputed_value_counts)


# In[174]:


# Verify the imputation:
imputed_value_counts = df['Computed_Smoking_Status'].value_counts(dropna=False,normalize=True) # 
print("Distribution after imputation:\n", imputed_value_counts)


# In[175]:


# Create a mapping dictionary:
smoking_Status_mapping = {1: 'current_smoker_every_day',
                           2: 'current_smoker_some_days',
                           3: 'former_smoker',
                           4: 'never_smoked'
                          }

# Apply the mapping to the smoking_Status_mapping column:
df['Computed_Smoking_Status'] = df['Computed_Smoking_Status'].map(smoking_Status_mapping)

# Rename the column:
df.rename(columns={'Computed_Smoking_Status': 'smoking_status'}, inplace=True)


# In[176]:


#view column counts & percentage:
value_counts_with_percentage(df, 'smoking_status')


# ### **Column 21: Binge_Drinking_Calculated_Variable**<a id='Column_21_Binge_Drinking_Calculated_Variable'></a>	
# [Contents](#Contents)

# In[177]:


#view column counts & percentage:
value_counts_with_percentage(df, 'Binge_Drinking_Calculated_Variable')


# **Binge_Drinking_Calculated_Variable:**
# * 1: 'no',
# * 2: 'yes',
# * 9: 'dont_know_refused_missing'
# 
# so for 9 let's convert to nan:

# In[178]:


#Replace 7 and 9 with NaN:
df['Binge_Drinking_Calculated_Variable'].replace([9], np.nan, inplace=True)
df.Binge_Drinking_Calculated_Variable.value_counts(dropna=False)


# In[179]:


# Calculate the distribution of existing values:
value_counts = df['Binge_Drinking_Calculated_Variable'].value_counts(normalize=True, dropna=True)
print("Original Binge_Drinking_Calculated_Variable:\n", value_counts)


# In[180]:


# Function to impute missing values based on distribution:
def impute_missing(row):
    if pd.isna(row['Binge_Drinking_Calculated_Variable']):
        return np.random.choice(value_counts.index, p=value_counts.values)
    else:
        return row['Binge_Drinking_Calculated_Variable']


# In[181]:


# Apply the imputation function:
df['Binge_Drinking_Calculated_Variable'] = df.apply(impute_missing, axis=1)


# In[182]:


# Verify the imputation:
imputed_value_counts = df['Binge_Drinking_Calculated_Variable'].value_counts(dropna=False) # normalize=True
print("Distribution after imputation:\n", imputed_value_counts)


# In[183]:


# Verify the imputation:
imputed_value_counts = df['Binge_Drinking_Calculated_Variable'].value_counts(dropna=False,normalize=True) # 
print("Distribution after imputation:\n", imputed_value_counts)


# In[184]:


# Create a mapping dictionary:
binge_drinking_status = {1: 'no',
                           2: 'yes'
                          }

# Apply the mapping to the binge_drinking_status column:
df['Binge_Drinking_Calculated_Variable'] = df['Binge_Drinking_Calculated_Variable'].map(binge_drinking_status)

# Rename the column:
df.rename(columns={'Binge_Drinking_Calculated_Variable': 'binge_drinking_status'}, inplace=True)


# In[185]:


#view column counts & percentage:
value_counts_with_percentage(df, 'binge_drinking_status')


# ### **Column 22: How_Much_Time_Do_You_Sleep**<a id='Column_22_How_Much_Time_Do_You_Sleep'></a>
# [Contents](#Contents)

# In[186]:


#view column counts & percentage:
value_counts_with_percentage(df, 'How_Much_Time_Do_You_Sleep')


# In[187]:


def categorize_sleep_hours(df, column_name):
    # Define the mapping dictionary for known values
    sleep_mapping = {
        77: 'dont_know',
        99: 'refused_to_answer',
        np.nan: 'missing'
    }
    
    # Categorize hours of sleep
    for hour in range(0, 4):
        sleep_mapping[hour] = 'very_short_sleep_0_to_3_hours'
    for hour in range(4, 6):
        sleep_mapping[hour] = 'short_sleep_4_to_5_hours'
    for hour in range(6, 9):
        sleep_mapping[hour] = 'normal_sleep_6_to_8_hours'
    for hour in range(9, 11):
        sleep_mapping[hour] = 'long_sleep_9_to_10_hours'
    for hour in range(11, 25):
        sleep_mapping[hour] = 'very_long_sleep_11_or_more_hours'

    # Map the values to their categories
    df['sleep_category'] = df[column_name].map(sleep_mapping)

    return df


# In[188]:


# Apply the function to categorize sleep hours
df = categorize_sleep_hours(df, 'How_Much_Time_Do_You_Sleep')


# In[189]:


#view column counts & percentage:
value_counts_with_percentage(df, 'sleep_category')


# In[190]:


#Replace 7 and 9 with NaN:
#df['sleep_category'].replace(['dont_know', 'refused_to_answer'], np.nan, inplace=True)
df['sleep_category'].replace(['missing', 'dont_know','refused_to_answer'], np.nan, inplace=True)
df.sleep_category.value_counts(dropna=False)


# In[191]:


# Calculate the distribution of existing values:
value_counts = df['sleep_category'].value_counts(normalize=True, dropna=True)
print("Original sleep_category:\n", value_counts)


# In[192]:


# Function to impute missing values based on distribution:
def impute_missing(row):
    if pd.isna(row['sleep_category']):
        return np.random.choice(value_counts.index, p=value_counts.values)
    else:
        return row['sleep_category']


# In[193]:


# Apply the imputation function:
df['sleep_category'] = df.apply(impute_missing, axis=1)


# In[194]:


# Verify the imputation:
imputed_value_counts = df['sleep_category'].value_counts(dropna=False) # normalize=True
print("Distribution after imputation:\n", imputed_value_counts)


# In[195]:


# Verify the imputation:
imputed_value_counts = df['sleep_category'].value_counts(dropna=False,normalize=True) # 
print("Distribution after imputation:\n", imputed_value_counts)


# In[196]:


#view column counts & percentage:
value_counts_with_percentage(df, 'sleep_category')


# ### **Column 23: Computed_number_of_drinks_of_alcohol_beverages_per_week**<a id='Column_23_Computed_number_of_drinks_of_alcohol_beverages_per_week'></a>
# [Contents](#Contents)

# In[197]:


#view column counts & percentage:
value_counts_with_percentage(df, 'Computed_number_of_drinks_of_alcohol_beverages_per_week')


# In[198]:


# Divide by 100 to get the number of drinks per week
df['drinks_per_week'] = df['Computed_number_of_drinks_of_alcohol_beverages_per_week'] / 100


# In[199]:


# Define the function to categorize the drink consumption
def categorize_drinks(drinks_per_week):
    #if drinks_per_week == 0:
        #return 'did_not_drink'
    if drinks_per_week == 99900 / 100:
        return 'do_not_know'
    elif 0.01 <= drinks_per_week <= 1:
        return 'very_low_consumption_0.01_to_1_drinks'
    elif 1.01 <= drinks_per_week <= 5:
        return 'low_consumption_1.01_to_5_drinks'
    elif 5.01 <= drinks_per_week <= 10:
        return 'moderate_consumption_5.01_to_10_drinks'
    elif 10.01 <= drinks_per_week <= 20:
        return 'high_consumption_10.01_to_20_drinks'
    elif drinks_per_week > 20:
        return 'very_high_consumption_more_than_20_drinks'
    else:
        return 'did_not_drink'


# In[200]:


# Apply the categorization function
df['drinks_category'] = df['drinks_per_week'].apply(categorize_drinks)


# In[201]:


#view column counts & percentage:
value_counts_with_percentage(df, 'drinks_category')


# In[202]:


#Replace 7 and 9 with NaN:
df['drinks_category'].replace(['do_not_know'], np.nan, inplace=True)
df.drinks_category.value_counts(dropna=False)


# In[203]:


# Calculate the distribution of existing values:
value_counts = df['drinks_category'].value_counts(normalize=True, dropna=True)
print("Original drinks_category:\n", value_counts)


# In[204]:


# Function to impute missing values based on distribution:
def impute_missing(row):
    if pd.isna(row['drinks_category']):
        return np.random.choice(value_counts.index, p=value_counts.values)
    else:
        return row['drinks_category']


# In[205]:


# Apply the imputation function:
df['drinks_category'] = df.apply(impute_missing, axis=1)


# In[206]:


# Verify the imputation:
imputed_value_counts = df['drinks_category'].value_counts(dropna=False) # normalize=True
print("Distribution after imputation:\n", imputed_value_counts)


# In[207]:


# Verify the imputation:
imputed_value_counts = df['drinks_category'].value_counts(dropna=False,normalize=True) # 
print("Distribution after imputation:\n", imputed_value_counts)


# In[208]:


#Final check after imputation:
value_counts_with_percentage(df, 'drinks_category')


# ## **Dropping unnecessary columns**<a id='Dropping_unnecessary_columns'></a>
# [Contents](#Contents)

# In[209]:


#Here, let's drop the unnecessary colums:
columns_to_drop = ['Imputed_Age_value_collapsed_above_80', 'Reported_Weight_in_Pounds', 
                   'Reported_Height_in_Feet_and_Inches', 'Leisure_Time_Physical_Activity_Calculated_Variable',
                  'Smoked_at_Least_100_Cigarettes', 'Computed_number_of_drinks_of_alcohol_beverages_per_week',
                  'How_Much_Time_Do_You_Sleep', 'drinks_per_week']
df = df.drop(columns=columns_to_drop)


# ## **Review the final structre of the cleaned dataframe**<a id='Review_final_structure_of_the_cleaned_dataframe'></a>
# [Contents](#Contents)

# In[210]:


#now, let's look at the shape of df:
shape = df.shape
print("Number of rows:", shape[0], "\nNumber of columns:", shape[1])


# In[211]:


summarize_df(df)


# Awesome, there's no missing data. **So, as we can see above. we cleaned the data, removed missing data and still maintained the size of the dataset "rows"**

# ## **Saving the clean dataframe**<a id='Saving_the_cleaned_dataframe'></a>
# [Contents](#Contents)

# In[212]:


output_file_path = "./brfss2022_data_wrangling_output.csv"

df.to_csv(output_file_path, index=False)


# In[ ]:





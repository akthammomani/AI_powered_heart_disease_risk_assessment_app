# Exploratory data analysis: AI-Powered Heart Disease Risk Assessment

![EDA](https://github.com/akthammomani/AI_powered_heart_disease_risk_assessment_app/assets/67468718/713fc503-ad47-4b4d-812e-6f52ddcec40b)

## **Introduction**

Welcome to the Exploratory Data Analysis (EDA) notebook for our heart disease prediction project. This notebook serves as a critical step in our data science workflow, aimed at uncovering insights and patterns within our dataset that will guide our predictive modeling efforts.

In this notebook, we will:

* Validate the Dataset: Ensure the data is clean, consistent, and ready for analysis.
* Explore Feature Distributions: Analyze the distribution of various features in relation to heart disease.
* Convert Categorical Data: Transform categorical features into numeric format using CatBoost encoding for better analysis and modeling.
* Analyze Correlations: Examine both linear and non-linear relationships between features and the target variable (heart disease) using Pearson correlation and mutual information.
* Feature Selection: Identify and select key features that have the most predictive power for heart disease.
  
These steps will help us understand the data better, reveal important relationships, and prepare the data for building robust predictive models.


## **Dataset**

The dataset used in this Exploratory Data Analysis (EDA) notebook is the result of a comprehensive data wrangling process. Data wrangling is a crucial step in the data science workflow, involving the transformation and preparation of raw data into a more usable format. The main tasks performed during data wrangling included:

* Dealing with missing data
* Data mapping
* Data cleaning
* Feature engineering
  
These steps ensured that the dataset is well-prepared for analysis and modeling, enabling us to build reliable and robust models for heart disease prediction.

## **Heart Disease related features**

After several days of research and analysis of the dataset's features, we have identified the following key features for heart disease assessment:

* **Target Variable (Dependent Variable):**
    * Heart_disease: "Ever_Diagnosed_with_Angina_or_Coronary_Heart_Disease"
* **Demographics:**
    * Gender: Are_you_male_or_female
    * Race: Computed_race_groups_used_for_internet_prevalence_tables
    * Age: Imputed_Age_value_collapsed_above_80
* **Medical History:**
    * General_Health
    * Have_Personal_Health_Care_Provider
    * Could_Not_Afford_To_See_Doctor
    * Length_of_time_since_last_routine_checkup
    * Ever_Diagnosed_with_Heart_Attack
    * Ever_Diagnosed_with_a_Stroke
    * Ever_told_you_had_a_depressive_disorder
    * Ever_told_you_have_kidney_disease
    * Ever_told_you_had_diabetes
    * Reported_Weight_in_Pounds
    * Reported_Height_in_Feet_and_Inches
    * Computed_body_mass_index_categories
    * Difficulty_Walking_or_Climbing_Stairs
    * Computed_Physical_Health_Status
    * Computed_Mental_Health_Status
    * Computed_Asthma_Status
* **Life Style:**
    * Leisure_Time_Physical_Activity_Calculated_Variable
    * Smoked_at_Least_100_Cigarettes
    * Computed_Smoking_Status
    * Binge_Drinking_Calculated_Variable
    * Computed_number_of_drinks_of_alcohol_beverages_per_week
    * Exercise_in_Past_30_Days
    * How_Much_Time_Do_You_Sleep

## **Converting features data type**

In pandas, the object data type is used for text or mixed data. When a column contains categorical data, it's often beneficial to explicitly convert it to the category data type. Here are some reasons why:

**Benefits of Converting to Categorical Type:**
* Memory Efficiency: Categorical data types are more memory efficient. Instead of storing each unique string separately, pandas stores the categories and uses integer codes to represent the values.
* Performance Improvement: Operations on categorical data can be faster since pandas can make use of the underlying integer codes.
* Explicit Semantics: Converting to category makes the data's categorical nature explicit, improving code readability and reducing the risk of treating categorical data as continuous.






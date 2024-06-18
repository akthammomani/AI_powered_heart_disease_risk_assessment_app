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

## **Analyzing categorical feature distributions against a target variable**

In data analysis, understanding the distribution of categorical features in relation to a target variable is crucial for gaining insights into the data. One effective way to achieve this is by using horizontal stacked bar charts. These visualizations allow us to see how different categories of a feature are distributed across the levels of the target variable, providing a clear view of relationships and patterns within the data.

### **Heart Disease: Target Variable**

![heart_disease_distribution](https://github.com/akthammomani/AI_powered_heart_disease_risk_assessment_app/assets/67468718/2334384d-f63f-4de3-b005-b503f5793cd9)
**Distribution Analysis**
* There is a significant imbalance between the two categories.
* A large majority of individuals do not have heart disease `418.3K`, while a much smaller number have heart disease `26.8K`.
* This imbalance can be visually observed in the chart, where the green bar is substantially longer than the red bar.

**Imbalance Issue**
* Model Bias: When training a classification model on this imbalanced dataset, the model might become biased towards predicting the majority class (No heart disease) more frequently because it is seen more often in the training data.
* Performance Metrics: Common performance metrics like accuracy can be misleading in imbalanced datasets. For instance, a model that always predicts "No heart disease" will have high accuracy because the majority class is well represented. However, this model would fail to correctly identify individuals with heart disease, which is critical for healthcare applications.
* Recall and Precision: Metrics such as recall (sensitivity) and precision are more informative in this context. Recall measures the ability to identify true positive cases (heart disease), while precision measures the accuracy of positive predictions. In an imbalanced dataset, a model might have low recall for the minority class (heart disease) even if it has high accuracy overall.

**Strategy to Address Imbalance**
The `BalancedRandomForestClassifier` from the `imbalanced-ensemble` library effectively handles class imbalance by using bootstrapped sampling to balance the dataset, ensuring robust classification of minority classes. It enhances model performance by focusing on underrepresented data, making it ideal for imbalanced datasets like heart disease prediction.


### **Heart Disease vs Gender**

![heart_disease_vs_gender](https://github.com/akthammomani/AI_powered_heart_disease_risk_assessment_app/assets/67468718/8f8c2858-d75a-4e46-ad15-4355b7a67322)

**Distribution Analysis**

* The majority of individuals with heart disease are male `15.5K`, followed by female `11.3K`.
* There are very few nonbinary individuals with heart disease `15 individuals`.
* The significant difference in the number of heart disease cases among males and females compared to nonbinary individuals highlights a noticeable imbalance.

### **Heart Disease vs Race**

![heart_disease_vs_race](https://github.com/akthammomani/AI_powered_heart_disease_risk_assessment_app/assets/67468718/50c3a545-6613-4ce5-810a-36d2408d019a)

**Distribution Analysis**

* The largest group with heart disease is "White Only, Non-Hispanic," with `22.2K` individuals.
* Smaller groups, such as "Native Hawaiian or Other Pacific Islander Only, Non-Hispanic" and "Asian Only, Non-Hispanic," have very few individuals with heart disease (`100` and `300` individuals, respectively).
* There is a notable imbalance in the number of heart disease cases across different racial categories, with significantly fewer cases in minority groups.

### **Heart Disease vs General Health**

![general_health](https://github.com/akthammomani/AI_powered_heart_disease_risk_assessment_app/assets/67468718/70b86a26-c021-4dc7-a5d1-77be326adbee)

**Distribution Analysis**

* The highest number of individuals with heart disease falls into the "Good" health category `8.9K`, followed by the "Fair" category `7.9K`.
* Both "Very Good" and "Poor" health categories have the same number of individuals with heart disease `4.5K`.
* The "Excellent" health category has the fewest individuals with heart disease `1.0K`.
* There is a noticeable distribution of heart disease cases across different general health categories, with the highest incidence in individuals who self-report as having "Good" or "Fair" health.

### **Heart Disease vs Health Care Provider**

![health_care_provider](https://github.com/akthammomani/AI_powered_heart_disease_risk_assessment_app/assets/67468718/ae7f9ae8-58f4-42a8-a83a-bbaa5c8899b4)

**Distribution Analysis**

* The highest number of individuals with heart disease is in the "More Than One" health care provider category `13.5K`.
* The "Yes, Only One" category also has a significant number of individuals with heart disease `12.2K`.
* The "No" health care provider category has the fewest individuals with heart disease `1.1K`.
* This distribution suggests that individuals with multiple health care providers or at least one provider are more likely to have heart disease compared to those with no health care provider.

### **Heart Disease vs Doctor availability**

![see_doctor](https://github.com/akthammomani/AI_powered_heart_disease_risk_assessment_app/assets/67468718/9f2fe2b2-4987-47a3-8cf6-837eb91d6ec1)

**Distribution Analysis**

* The majority of individuals with heart disease fall into the category of those who could afford to see a doctor `24.5K`.
* A smaller number of individuals with heart disease could not afford to see a doctor `2.3K`.
* This distribution indicates that even among those with heart disease, most individuals could afford to see a doctor, suggesting access to healthcare does not completely mitigate the risk of heart disease.
* However, the presence of heart disease in individuals who could not afford to see a doctor highlights a potential issue with access to preventive care or treatment.


















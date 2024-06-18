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

### **Heart Disease vs Routine Checkup**

![routine_checkup](https://github.com/akthammomani/AI_powered_heart_disease_risk_assessment_app/assets/67468718/f9d1958e-aa43-4dec-92de-cca116093e81)

**Distribution Analysis**

* The majority of individuals with heart disease had a routine checkup within the past year `24.9K`. This indicates that individuals with heart disease are more likely to have had recent medical attention.
* There are significantly fewer individuals with heart disease who had a routine checkup in the past 2 years `1.1K` and past 5 years `0.5K`.
* A very small number of individuals with heart disease reported never having a routine checkup `0.1K` or having their last checkup more than 5 years ago `0.3K`.
* This distribution suggests that even those with recent medical checkups are at risk of heart disease, highlighting the importance of regular monitoring and early detection. However, individuals with infrequent or no checkups are less likely to be diagnosed, possibly due to lower health awareness or access to healthcare.

### **Heart Disease vs Heart Attack**

![heart_attack](https://github.com/akthammomani/AI_powered_heart_disease_risk_assessment_app/assets/67468718/6d06b6d8-ee84-4586-b367-0af38a4b3808)

**Distribution Analysis**

* A significant number of individuals with heart disease have also been diagnosed with a heart attack `12.0K`. This indicates a strong correlation between a previous heart attack and the presence of heart disease.
* There are also a substantial number of individuals with heart disease who have not been diagnosed with a heart attack `14.8K`. This highlights that heart disease can develop without a prior heart attack diagnosis.
* The distribution suggests that while a prior heart attack is a significant indicator of heart disease, many individuals with heart disease have no history of a heart attack, emphasizing the need for comprehensive cardiovascular risk assessments beyond just heart attack history.

### **Heart Disease vs Stroke**

![stroke](https://github.com/akthammomani/AI_powered_heart_disease_risk_assessment_app/assets/67468718/b3b4342a-2af1-43c6-9e74-d9069337f92c)

**Distribution Analysis**

* A significant number of individuals with heart disease have also been diagnosed with a stroke `4.4K`. This indicates a notable correlation between a previous stroke and the presence of heart disease.
* A larger number of individuals with heart disease have not been diagnosed with a stroke `22.4K`. This shows that heart disease can occur independently of a stroke diagnosis.
* The distribution suggests that while a prior stroke is a significant risk factor for heart disease, many individuals with heart disease do not have a history of stroke. This highlights the importance of a comprehensive cardiovascular risk assessment, considering various risk factors beyond just stroke history.

### **Heart Disease vs Kidney Disease**

![kidney](https://github.com/akthammomani/AI_powered_heart_disease_risk_assessment_app/assets/67468718/01a61ad3-ca34-46f2-ab93-f81f4c87849f)

**Distribution Analysis**

* A notable number of individuals with heart disease have also been diagnosed with kidney disease `4.5K`. This indicates a correlation between kidney disease and heart disease.
* A larger number of individuals with heart disease have not been diagnosed with kidney disease `22.3K`. This shows that heart disease frequently occurs independently of kidney disease.
* The distribution suggests that while kidney disease is a significant risk factor for heart disease, the majority of individuals with heart disease do not have a history of kidney disease. This underscores the importance of evaluating multiple risk factors for heart disease, including but not limited to kidney disease history.

### **Heart Disease vs Diabetes**

![diabetes](https://github.com/akthammomani/AI_powered_heart_disease_risk_assessment_app/assets/67468718/cd558f4d-017d-408f-acb8-77c575e03bd3)

**Distribution Analysis**

* The highest number of individuals with heart disease are in the category of those who do not have diabetes `16.5K`. This indicates that heart disease is prevalent even among those without a diabetes diagnosis.
* A significant number of individuals with heart disease have been diagnosed with diabetes `9.3K`. This highlights the strong correlation between diabetes and heart disease.
* A smaller number of individuals with heart disease have prediabetes `0.9K` or had diabetes during pregnancy `0.1K`, suggesting these conditions are less common among those with heart disease compared to a full diabetes diagnosis.
* The distribution emphasizes the importance of monitoring and managing diabetes as a critical risk factor for heart disease. However, the presence of heart disease in individuals without diabetes underscores the multifactorial nature of cardiovascular risk.

### **Heart Disease vs BMI**

![bmi](https://github.com/akthammomani/AI_powered_heart_disease_risk_assessment_app/assets/67468718/b44f5d85-c4b7-40d4-b4b2-c4c98315f9b3)

**Distribution Analysis**

* The highest number of individuals with heart disease falls into the obese category `10.6K`, indicating a strong correlation between obesity and heart disease.
* The overweight category also has a significant number of individuals with heart disease `9.6K`, further highlighting the relationship between higher BMI and heart disease risk.
* Individuals with a normal weight `6.2K` and underweight `0.4K` have fewer cases of heart disease, suggesting that maintaining a normal weight may be associated with a lower risk of heart disease.
* This distribution underscores the importance of managing body weight as a critical factor in reducing the risk of heart disease, with obesity and overweight being key areas of concern.

### **Heart Disease vs Difficulty Walking or Climbing**

![climbing](https://github.com/akthammomani/AI_powered_heart_disease_risk_assessment_app/assets/67468718/1e94935c-c1f2-4947-a992-13ae8a3d3d73)

**Distribution Analysis**

* A notable number of individuals with heart disease report having difficulty walking or climbing stairs `10.8K`. This indicates a strong association between mobility issues and the presence of heart disease.
* A slightly higher number of individuals with heart disease do not report difficulty walking or climbing stairs `16.0K`. This shows that heart disease can occur even in those without significant mobility issues.
* The distribution suggests that difficulty in walking or climbing stairs is a significant risk factor for heart disease, but it also highlights that heart disease is present in a considerable number of individuals without mobility challenges. This underscores the importance of comprehensive cardiovascular risk assessments that consider a variety of health factors.


### **Heart Disease vs Physical Health Status**

![physical_health](https://github.com/akthammomani/AI_powered_heart_disease_risk_assessment_app/assets/67468718/69434e49-932f-43ff-aafb-15842ea2a1fe)

**Distribution Analysis**

* The highest number of individuals with heart disease report having zero days of not feeling good physically `11.6K`. This indicates that a significant portion of individuals with heart disease perceive their physical health as generally good.
* A notable number of individuals with heart disease report having 14 or more days of not feeling good physically `8.4K`. This suggests a correlation between prolonged periods of poor physical health and heart disease.
* Individuals reporting 1 to 13 days of not feeling good physically account for `6.8K` cases of heart disease.
* The distribution highlights the importance of considering physical health status in assessing the risk of heart disease. While many individuals with heart disease report good physical health, there is a substantial group experiencing frequent poor physical health days, which may be an indicator of underlying issues.


### **Heart Disease vs Mental Health Status**

![mental_health](https://github.com/akthammomani/AI_powered_heart_disease_risk_assessment_app/assets/67468718/628ac1df-f28f-4c69-ae6e-65eb5f1e4e90)

**Distribution Analysis**

* The highest number of individuals with heart disease report having zero days of not feeling good mentally `16.7K`. This suggests that many individuals with heart disease perceive their mental health as generally good.
* A significant number of individuals with heart disease report having 1 to 13 days of not feeling good mentally `5.4K`. This indicates a noticeable correlation between moderate periods of poor mental health and the presence of heart disease.
* Individuals reporting 14 or more days of not feeling good mentally account for `4.7K` cases of heart disease. This suggests that prolonged periods of poor mental health are also a factor among individuals with heart disease.
* The distribution highlights the importance of considering mental health status in assessing the risk of heart disease. While many individuals with heart disease report good mental health, there is a substantial group experiencing frequent poor mental health days, indicating the need for comprehensive health evaluations.


### **Heart Disease vs Asthma**

![asthma](https://github.com/akthammomani/AI_powered_heart_disease_risk_assessment_app/assets/67468718/a210a039-189d-40ae-bdbc-1234dae26f33)

**Distribution Analysis**

* The highest number of individuals with heart disease falls into the "Never Asthma" category `21.5K`. This suggests that a large portion of individuals with heart disease do not have a history of asthma.
* A smaller number of individuals with heart disease have current asthma `4.1K`, indicating a correlation between ongoing asthma and heart disease.
* The smallest number of individuals with heart disease are those with former asthma `1.2K`. This suggests that having had asthma in the past is less common among individuals with heart disease compared to never having had asthma or currently having asthma.
* The distribution highlights the importance of considering asthma status in assessing the risk of heart disease. While many individuals with heart disease have never had asthma, there is still a significant group with current asthma, indicating the need for careful monitoring and management of both conditions.

### **Heart Disease vs Smoking status**

![smoking_status](https://github.com/akthammomani/AI_powered_heart_disease_risk_assessment_app/assets/67468718/6a695f99-9d2d-4b57-a19e-aa6b8ed5df9d)

**Distribution Analysis**

* The highest number of individuals with heart disease are those who have never smoked `12.1K`. This may reflect the larger population size of never smokers.
* Former smokers have a high number of heart disease cases `11.1K`. This suggests that the health effects of smoking may persist even after quitting, leading to higher rates of heart disease among former smokers.
* Current smokers who smoke every day also have a notable number of heart disease cases `2.7K`, indicating that ongoing smoking significantly contributes to heart disease risk.
* Interestingly, current smokers who smoke only on some days have the fewest heart disease cases `0.9K`. This might be due to the smaller size of this subgroup or underreporting.

**Why Former Smokers Have Higher Cases Compared to Current Smokers (Some Days)?**

* Long-Term Effects of Smoking: Former smokers may have smoked for many years before quitting, leading to cumulative damage to their cardiovascular system. The adverse effects of prolonged smoking can persist long after quitting, increasing the risk of heart disease.
* Health Improvements: Current smokers who smoke only on some days might have a lower overall exposure to smoking-related toxins compared to those who smoked regularly for years before quitting.
* Population Size: The former smoker category likely includes a larger and more diverse group of individuals than the current smokers (some days) category, which might be relatively smaller and less representative of heavy, long-term smokers.
  
The distribution highlights the importance of considering smoking history in assessing heart disease risk. Even after quitting, former smokers continue to face a high risk, underscoring the long-term health impacts of smoking. Current smokers, especially those who smoke daily, also face significant risks, emphasizing the need for smoking cessation programs and ongoing monitoring of cardiovascular health.


### **Heart Disease vs Binge Drinking Status**

![binge](https://github.com/akthammomani/AI_powered_heart_disease_risk_assessment_app/assets/67468718/acdda6f1-7e10-491c-8cbc-83801129f9d0)

**Distribution Analysis**

* The majority of individuals with heart disease do not engage in binge drinking `24.8K`. This suggests that while binge drinking is a risk factor, many individuals with heart disease do not exhibit this behavior.
* A smaller number of individuals with heart disease report engaging in binge drinking `2.0K`. This indicates that binge drinking is associated with heart disease, but it is less prevalent among heart disease patients compared to those who do not binge drink.
* The distribution highlights the importance of considering alcohol consumption patterns in assessing the risk of heart disease. While binge drinking is a significant risk factor, it is not the sole determinant of heart disease, as a substantial number of heart disease cases occur in individuals who do not binge drink. This underscores the multifactorial nature of heart disease risk and the need for comprehensive health assessments that include lifestyle factors such as alcohol consumption.


### **Heart Disease vs Exercise Status**

![exercise](https://github.com/akthammomani/AI_powered_heart_disease_risk_assessment_app/assets/67468718/bcd7ed5f-ec3f-4478-8455-c8dfc9ea691d)

**Distribution Analysis**

* A significant number of individuals with heart disease report having exercised in the past 30 days `16.8K`. This suggests that exercise, while beneficial, does not entirely prevent the occurrence of heart disease, possibly due to other overriding risk factors.
* A notable number of individuals with heart disease report not having exercised in the past 30 days `10.0K`. This indicates a correlation between lack of exercise and the presence of heart disease.
* The distribution highlights the importance of regular physical activity as a component of heart disease prevention. However, it also underscores that exercise alone is not sufficient to mitigate all risks associated with heart disease, emphasizing the need for a holistic approach to cardiovascular health that includes diet, lifestyle changes, and regular medical checkups.


### **Heart Disease vs Age Category**

![age](https://github.com/akthammomani/AI_powered_heart_disease_risk_assessment_app/assets/67468718/497cc521-e176-4940-b34f-f954908dbdbf)

**Distribution Analysis**

* The highest number of individuals with heart disease are in the age category of 70 to 74 `4.9K`, followed closely by the 80 or older category `5.7K`, and 75 to 79 category `4.4K`. This suggests a strong correlation between advanced age and the presence of heart disease.
* The number of individuals with heart disease generally increases with age, peaking in the older age categories. This indicates that age is a significant risk factor for heart disease.
* There are relatively few individuals with heart disease in the younger age categories (18 to 24 and 25 to 29), highlighting that while younger individuals can have heart disease, it is less common compared to older age groups.
* The distribution underscores the importance of age as a critical factor in heart disease risk assessment. It suggests that preventive measures and monitoring should be more rigorous as individuals age, particularly for those over 65.


### **Heart Disease vs Sleep Category**

![sleep](https://github.com/akthammomani/AI_powered_heart_disease_risk_assessment_app/assets/67468718/1f887af9-a60e-4656-8741-763fa06cf748)

**Distribution Analysis**

* The highest number of individuals with heart disease are in the normal sleep category (6 to 8 hours) with `19.5K` individuals. This suggests that many individuals with heart disease report getting a standard amount of sleep.
* There are fewer individuals with heart disease in the very short sleep (0 to 3 hours) and very long sleep (11 or more hours) categories, each with `0.6K` individuals. This indicates that extreme sleep durations are less common among those with heart disease, but they are still present.
* Short sleep (4 to 5 hours) has `3.3K` individuals with heart disease, showing a significant correlation between insufficient sleep and heart disease.
* Long sleep (9 to 10 hours) includes `2.8K` individuals with heart disease, suggesting that extended sleep duration is also associated with heart disease, though to a lesser extent than normal sleep duration.
* The distribution highlights the complex relationship between sleep duration and heart disease. While normal sleep duration is common among those with heart disease, both insufficient and excessive sleep are also important factors to consider in cardiovascular risk assessments. This underscores the importance of promoting healthy sleep habits as part of overall heart health.


### **Heart Disease vs Drinking Status**

![drinks](https://github.com/akthammomani/AI_powered_heart_disease_risk_assessment_app/assets/67468718/99a16ce4-349e-4ce7-92a8-aa75e14c96cd)

**Distribution Analysis**

* The highest number of individuals with heart disease fall into the "Did Not Drink" category with `15.9K` individuals. This suggests that abstaining from alcohol is common among those with heart disease, possibly due to health reasons or pre-existing conditions.
* Very low alcohol consumption (0.01 to 1 drinks) is the second most common category among those with heart disease, with `4.0K` individuals. This indicates that minimal alcohol consumption is still present among those with heart disease.
* Low consumption (1.01 to 5 drinks) includes `3.4K` individuals with heart disease, highlighting that moderate drinking is also observed among this population.
* Moderate consumption (5.01 to 10 drinks) and high consumption (10.01 to 20 drinks) have fewer cases of heart disease with `1.7K` and `1.0K` individuals respectively.
* Very high consumption (more than 20 drinks) is the least common among those with heart disease, with `0.7K` individuals.
* The distribution suggests that while many individuals with heart disease either do not drink or consume very little alcohol, moderate to high consumption is less common among this group. This emphasizes the importance of considering alcohol consumption habits in the context of heart disease risk and highlights the potential benefits of moderate or no alcohol consumption for heart health.



























































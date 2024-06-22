# Modeling: AI-Powered Heart Disease Risk Assessment

<center>
    <img src="https://github.com/akthammomani/AI_powered_heart_disease_risk_assessment_app/assets/67468718/e457d8ec-3f3d-4580-ae6e-bac3329ba681" alt="modeling">
</center>

## **Introduction**

In this notebook, we will be fitting and evaluating multiple machine learning models to classify heart disease:

* Logistic Regression
* Random Forest
* XGBoost
* LightGBM
* Balanced Bagging
* Easy Ensemble
* Balanced Random Forest	
* Balanced Bagging (LightGBM): Balanced Bagging as a Wrapper and LightGBM as a base estimator
* Easy Ensemble (LightGBM): Easy Ensemble as a Wrapper and LightGBM as a base estimator	
  
Our goal is to accurately predict heart disease risk using these models. We will employ hyperparameter tuning with `Optuna` to optimize each model's performance. Additionally, we will leverage the `BalancedRandomForestClassifier`, `BalancedBaggingClassifier` and `EasyEnsembleClassifier` from the `imbalanced-learn library` to address class imbalance. These classifiers use bootstrapped sampling to balance the dataset, ensuring robust classification of minority classes. By focusing on underrepresented data, it enhances model performance, making it particularly suitable for imbalanced datasets like heart disease prediction.

Through this comprehensive approach, we aim to develop a reliable and effective model for heart disease risk assessment, contributing to better health outcomes.

## **Dataset**

The dataset used in this Modeling notebook is the result of a comprehensive data wrangling process. Data wrangling is a crucial step in the data science workflow, involving the transformation and preparation of raw data into a more usable format. The main tasks performed during data wrangling included:

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
 
## **Converting Features Data Type**

In pandas, the object data type is used for text or mixed data. When a column contains categorical data, it's often beneficial to explicitly convert it to the category data type. Here are some reasons why:

**Benefits of Converting to Categorical Type:**
* Memory Efficiency: Categorical data types are more memory efficient. Instead of storing each unique string separately, pandas stores the categories and uses integer codes to represent the values.
* Performance Improvement: Operations on categorical data can be faster since pandas can make use of the underlying integer codes.
* Explicit Semantics: Converting to category makes the data's categorical nature explicit, improving code readability and reducing the risk of treating categorical data as continuous.

## **Heart Disease: Target Variable**

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
The `BalancedRandomForestClassifier` or `BalancedBaggingClassifier` or `EasyEnsembleClassifier` from the `imbalanced-learn` library effectively handles class imbalance by using bootstrapped sampling to balance the dataset, ensuring robust classification of minority classes. It enhances model performance by focusing on underrepresented data, making it ideal for imbalanced datasets like heart disease prediction.

## **Categorical Encoding with Catboost**

Many machine learning algorithms require data to be numeric. Therefore, before training a model or calculating the correlation (Pearson) or mutual information (prediction power), we need to convert categorical data into numeric form. Various categorical encoding methods are available, and CatBoost is one of them. CatBoost is a target-based categorical encoder. It is a supervised encoder that encodes categorical columns according to the target value, supporting both binomial and continuous targets.

Target encoding is a popular technique used for categorical encoding. It replaces a categorical feature with average value of target corresponding to that category in training dataset combined with the target probability over the entire dataset. But this introduces a target leakage since the target is used to predict the target. Such models tend to be overfitted and don’t generalize well in unseen circumstances.

A CatBoost encoder is similar to target encoding, but also involves an ordering principle in order to overcome this problem of target leakage. It uses the principle similar to the time series data validation. The values of target statistic rely on the observed history, i.e, target probability for the current feature is calculated only from the rows (observations) before it.


## **Baseline Modeling**

Here, we will fit the the following models listed below and compare their performance both at the overall model level and at the class-specific level:

* Logistic Regression
* Random Forest
* XGBoost
* LightGBM
* Balanced Bagging
* Easy Ensemble
* Balanced Random Forest	
* Balanced Bagging (LightGBM): Balanced Bagging as a Wrapper and LightGBM as a base estimator
* Easy Ensemble (LightGBM): Easy Ensemble as a Wrapper and LightGBM as a base estimator

### **Class-specific level Metrics Comparison**

![baseline_models](https://github.com/akthammomani/AI_powered_heart_disease_risk_assessment_app/assets/67468718/b3c2483d-1b55-48f7-abc4-72d0e3da7bd2)

* **High Recall, Low Precision and F1 Score:**
  * All models have high recall but low precision and F1 scores. This indicates that they are good at identifying positive cases (patients with heart disease) but also tend to predict a significant number of false positives (patients incorrectly identified as having heart disease).
* **Balanced Bagging and Easy Ensemble:**
  * These models are designed to handle class imbalance by balancing the classes during training. As a result, they have high recall, meaning they capture most of the actual positive cases. However, the trade-off is lower precision, which leads to a lower F1 score.
In a medical context, high recall is crucial as it is important to identify as many true positive cases as possible, even at the cost of some false positives. Missing a true positive (false negative) could be more critical than having a false positive.
* **Using LightGBM as Base Estimator:**
  * When using LightGBM as the base estimator in Balanced Bagging and Easy Ensemble, the results show a similar pattern of high recall and low precision. However, these models have slightly better ROC AUC scores `(0.885894 and 0.885778, respectively)`, indicating a good balance between sensitivity and specificity.
  * LightGBM is a powerful gradient boosting framework known for its efficiency and performance, which helps in achieving better overall performance metrics.
  * When using Easy Ensemble as a wrapper and LightGBM as a base estimator, lightGBM **Recall has improved from `24.4% to 80.7%` and ROC AUC has improved from `88.4% to 88.6%` for class 1 (heart disease patients)**
* **Practical Implications:** For a heart disease classification task, where identifying patients with heart disease (true positives) is critical, high recall is generally more desirable, even at the cost of having more false positives. This is because:
  * High Recall: Ensures most patients with heart disease are identified, which is crucial for early intervention and treatment.
  * False Positives: While not ideal, they can be managed through follow-up testing and further medical evaluation.





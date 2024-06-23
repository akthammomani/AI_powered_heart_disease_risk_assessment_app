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
    * Gender
    * Race
    * Age_category
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
    * BMI
    * Difficulty_Walking_or_Climbing_Stairs
    * Physical_Health_Status
    * Mental_Health_Status
    * Asthma_Status
* **Life Style:**
    * Smoking_status
    * Binge_Drinking_status
    * Drinks_category
    * Exercise_in_Past_30_Days
    * Sleep_category
 
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

Here, we will fit the the following models listed below and compare their performance at the class-specific level:

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

* **High Recall, Low Precision and F1 Score:** The majority of models show poor recall for class 1 (patients with heart disease), except for Balanced Bagging, Easy Ensemble, Balanced Random Forest, and when these models are combined with LightGBM. This indicates that most models struggle to identify positive cases (patients with heart disease), resulting in a significant number of false negatives (patients incorrectly identified as not having heart disease).
* **Balanced Bagging and Easy Ensemble:**
  * Balanced Bagging and Easy Ensemble models, along with Balanced Random Forest, are designed to handle class imbalance by balancing the classes during training.
  * Performance:
    * They achieve higher recall for class 1, meaning they capture most of the actual positive cases.
    * The trade-off is typically lower precision, leading to a lower F1 score.
* **Medical Context Implication:** In a medical context, high recall is crucial as it is important to identify as many true positive cases as possible, even at the cost of some false positives. Missing a true positive (false negative) could be more critical than having a false positive.
* **Using LightGBM as Base Estimator:**
  * Performance with LightGBM:
    * When using LightGBM as the base estimator in Balanced Bagging and Easy Ensemble, the results show improved recall for class 1.
    * These models also have slightly better ROC AUC scores (0.885894 and 0.885778, respectively), indicating a good balance between sensitivity and specificity.
    * LightGBM is a powerful gradient boosting framework known for its efficiency and performance, which helps in achieving better overall performance metrics.
  * Improvement:
    * When using Easy Ensemble as a wrapper and LightGBM as a base estimator, the Recall for class 1 (heart disease patients) improves significantly from 24.4% (in standalone LightGBM) to 80.7%.
    * The ROC AUC improves from 88.4% to 88.6% for class 1, showing a better balance between correctly identifying true positives and minimizing false positives.
* **Practical Implications:** Heart Disease Classification Task:
  * Identifying patients with heart disease (true positives) is critical.
  * High recall is generally more desirable, even at the cost of having more false positives.
  * High Recall ensures most patients with heart disease are identified, which is crucial for early intervention and treatment.
  * False Positives, while not ideal, can be managed through follow-up testing and further medical evaluation.
* **Conclusion:**
Balanced Bagging, Easy Ensemble, and Balanced Random Forest models, particularly with LightGBM as the base estimator, provide a good balance between identifying true positives and maintaining a reasonable rate of false positives.
For a medical application such as heart disease prediction, these approaches ensure that most cases of heart disease are identified, enabling timely medical intervention, which is crucial for patient care.


## **Hyperparameter Tuning using Optuna**

```python
Best hyperparameters:
  {'n_estimators': 10,
'learning_rate': 0.1,
'boosting_type': 'gbdt',
'num_leaves': 104,
'max_depth': 10,
'min_child_samples': 24,
'subsample': 0.8437808863271848,
'colsample_bytree': 0.8,
'reg_alpha': 0,
'reg_lambda': 0.6}
```

## **Fitting Best Model - EasyEnsemble as a wrapper and LightGBM as a base estimator:**

![best_model](https://github.com/akthammomani/AI_powered_heart_disease_risk_assessment_app/assets/67468718/c1a8839e-cd87-47b0-88e3-18ba87ba9f47)

* Class 0:
  * The model has a high precision `(0.984788)` for class 0, indicating that when it predicts class 0, it is correct `98.48%` of the time.
  * The recall for class 0 is also reasonably high `(0.785166)`, meaning it correctly identifies `78.52%` of all actual class 0 instances.
  * The F1 score `(0.873720)` shows a good balance between precision and recall.
  * The ROC AUC for class 0 is `0.785166`, indicating good discriminative ability.
* Class 1:
  * The precision for class 1 is low `(0.197094)`, meaning many of the predicted class 1 instances are actually class 0.
  * However, the recall for class 1 is high `(0.813019)`, indicating the model is good at identifying actual class 1 instances.
  * The F1 score for class 1 is relatively low `(0.317274)`, suggesting a trade-off between precision and recall.
  * The ROC AUC for class 1 is the same as for class 0 `(0.883942)`, indicating overall good model performance in distinguishing between classes.
    
**Comparison to Separate and Combined Models:**
* LightGBM Alone:
  * LightGBM typically has strong performance due to its gradient boosting capabilities. It may achieve high accuracy and good precision/recall balances for both classes.
  * However, LightGBM alone might struggle with class imbalance, often resulting in lower recall for minority classes (class 1: Recall `24.4%`).
* EasyEnsemble Alone:
  * EasyEnsemble without LightGBM as the base estimator focuses on balancing the data using under-sampling and creating multiple models.
  * This approach improves recall for minority classes but might not achieve the high precision that LightGBM offers.
  * The combined approach of using EasyEnsemble with LightGBM leverages the strengths of both techniques, enhancing both precision and recall, particularly for the minority class.
* Combined (Tuned):
  * When tuned, EasyEnsemble with LightGBM as the base estimator provides a balanced approach to handle class imbalance.
  * The combined method shows improved recall for class 1 ` from `24.4% to 81.3%` while maintaining a good precision for class 0 `(0.984788)`.
  * This combination also ensures that the model has a robust overall performance as indicated by the ROC AUC of `0.883942`.
    
**Conclusion:**

Using EasyEnsemble with LightGBM as the base estimator, especially when hyperparameters are tuned, offers a comprehensive solution to handling class imbalance. It ensures high precision and recall for class 0 and significantly improves recall for class 1, although precision for class 1 remains a challenge. This combined approach outperforms using LightGBM or EasyEnsemble separately by effectively leveraging the strengths of both methods.

## **Tuned Best Model Features Importance Using SHAP**

![shap_analysis](https://github.com/akthammomani/AI_powered_heart_disease_risk_assessment_app/assets/67468718/e0bb8855-3504-4e49-8c8f-3e967c8fdb51)

**Summary of SHAP Summary Plot for Class 1 (Heart Disease Patients):**

Above SHAP summary plot shows the impact of each feature on the model's output for predicting heart disease (class 1). Each dot represents a SHAP value for a feature, with the color indicating the feature's value (red for high and blue for low). Here’s a detailed interpretation:

* **High Positive Impact:**
  * Age Category: Higher age (represented by red dots) increases the likelihood of heart disease. Age is a significant risk factor, with older individuals being more prone to heart disease.
  * Ever Diagnosed with Heart Attack: A history of heart attacks (red dots) greatly increases the likelihood of heart disease. This past medical history is a strong indicator of recurring or persistent heart issues.
* **Moderate Positive Impact:**
  * General Health: Poor general health (red dots) increases the likelihood of heart disease. Individuals with overall poor health are at higher risk.
  * Health Care Provider: Frequent visits to a healthcare provider (red dots) indicate a higher likelihood of heart disease, possibly due to ongoing health issues necessitating regular check-ups.
  * Ever Told You Had Diabetes: Being told by a healthcare provider that you have diabetes (red dots) increases the likelihood of heart disease. Diabetes is a well-known risk factor for cardiovascular diseases.
  * Gender: Certain gender-related factors (likely male, indicated by red dots) increase the likelihood of heart disease. Men generally have a higher risk of heart disease at a younger age compared to women.
  * Difficulty Walking or Climbing Stairs: Difficulty in these activities (red dots) indicates a higher risk of heart disease, possibly due to underlying cardiovascular issues.
  * Ever Told You Have Kidney Disease: A history of kidney disease (red dots) increases the likelihood of heart disease. Kidney disease can be associated with cardiovascular complications.
  * Length of Time Since Last Routine Checkup: A longer time since the last checkup (red dots) increases the likelihood of heart disease. Regular check-ups can help manage and prevent health issues.
  * Race: Certain racial factors (red dots) increase the likelihood of heart disease, highlighting the role of demographic and genetic factors.
  * Ever Diagnosed with a Stroke: A history of stroke (red dots) increases the likelihood of heart disease, as both share common risk factors.
  * Physical Health Status: Poor physical health (red dots) increases the likelihood of heart disease, reflecting the impact of overall physical well-being on heart health.
  * Smoking Status: Being a smoker (red dots) significantly increases the likelihood of heart disease. Smoking is a major risk factor for cardiovascular diseases.
  * BMI: Higher BMI (red dots) increases the likelihood of heart disease. Obesity is closely linked to cardiovascular risk.
  * Asthma Status: Higher severity of asthma (red dots) increases the risk, possibly due to the overall impact of chronic respiratory conditions on health.
  * Could Not Afford to See Doctor: Financial barriers to healthcare (red dots) increase the likelihood of heart disease, likely due to untreated health conditions.
  * Ever Told You Had a Depressive Disorder: A history of depressive disorder (red dots) increases the likelihood of heart disease, indicating a connection between mental and cardiovascular health.
  * Drinks Category: Higher alcohol consumption (red dots) is associated with an increased risk of heart disease. Excessive drinking can negatively impact heart health.
  * Binge Drinking Status: Higher frequency of binge drinking (red dots) increases the likelihood of heart disease, highlighting the adverse effects of excessive alcohol intake.
  * Mental Health Status: Poor mental health (red dots) increases the likelihood of heart disease, emphasizing the importance of mental well-being for heart health.  
* **Mixed Impact:**
  * Race: Certain racial factors have a mixed influence but can increase the likelihood of heart disease, showing the importance of considering demographic variables in health risk assessments.
  * Asthma Status: While generally having a moderate positive impact, higher severity of asthma (red dots) increases the risk of heart disease. This indicates that severe respiratory issues can contribute to cardiovascular risk.
* **Conclusion:**
The SHAP summary plot illustrates that various factors such as age, history of heart attacks, general health, and diabetes significantly impact the likelihood of heart disease. The analysis emphasizes the importance of regular health check-ups, managing chronic conditions, and addressing both physical and mental health to mitigate the risk of heart disease.







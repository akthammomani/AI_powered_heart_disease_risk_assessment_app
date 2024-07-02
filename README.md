# AI-Powered Heart Disease Risk Assessment App [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_red.svg)](https://ai-powered-heart-disease-assessment.streamlit.app/)
This project is a part of the Probability and Statistics for Artificial Intelligence (AAI-500) course in [the Applied Artificial Intelligence Master Program](https://onlinedegrees.sandiego.edu/masters-applied-artificial-intelligence/) at [the University of San Diego (USD)](https://www.sandiego.edu/). 

[streamlit app](https://ai-powered-heart-disease-assessment.streamlit.app/)
-- **Project Status: Completed**

## **Introduction**:

The AI-Powered Heart Disease Risk Assessment App is a comprehensive tool designed to empower individuals with personalized insights into their cardiovascular health. While the app can evaluate multiple health indicators, it primarily focuses on predicting the risk of heart disease. This includes key factors such as:
* Age
* Gender
* BMI
* Physical activity levels
* Smoking status
* Medical history (e.g., previous heart attacks, strokes, diabetes)

By providing a thorough analysis of users' heart health status, the app aims to help users understand their risk and take proactive steps to maintain a healthy heart.

## **Objectives**: 

The AI-Powered Heart Disease Risk Assessment App aims to provide users with tailored risk scores and actionable recommendations to help them mitigate their risk of heart disease. Through easy-to-understand assessments and preventive measures, all powered by advanced AI and modeling techniques, the app makes safeguarding your cardiovascular health accessible and straightforward.

## **Methods Used**

* Data Wrangling
* Exploratory Data Analysis (EDA)
* Data Visualization
* Handling Imbalanced Classification
* Feature Engineering and Encoding
* Correlation Analysis
* Hyperparameter Tuning

## **Technologies**

* **Python**: The main programming language used for the project.
* **Streamlit**: For developing and deploying the app using Streamlit Sharing.
* **HTML & CSS**: Web APP personalization.
* **CatBoost**: For encoding categorical features.
* **Pearson Correlation**: Used for measuring the linear relationship between features and the target variable, helping to identify the strength and direction of linear associations.
* **Mutual Information**: Used for measuring both linear and non-linear relationships between features and the target variable, providing insights into the dependency and relevance of features.
* **OPTUNA**: For hyperparameter tuning.
* **imbalanced-learn library**: To effectively handles class imbalance by using bootstrapped sampling to balance the dataset.
* **Logistic Regression, Random Forest, LightGBM, XGBoost**: Classification models.
* **SHAP**: Features Importance.

## **Repository Contents**: 
* [Data Wrangling and Pre-Processing Code](https://github.com/akthammomani/AI_powered_heart_disease_risk_assessment_app/tree/main/Notebooks/Data_wrangling_pre_processing)
* [Exploratory Data Analysis (EDA) Code](https://github.com/akthammomani/AI_powered_heart_disease_risk_assessment_app/tree/main/Notebooks/Exploratory_Data_Analysis)
* [Modeling Code](https://github.com/akthammomani/AI_powered_heart_disease_risk_assessment_app/tree/main/Notebooks/Modeling)
* [App Development](https://github.com/akthammomani/AI_powered_heart_disease_risk_assessment_app/tree/main/App)

## **Features:**

* **Personalized Heart Disease Risk Assessment**: Evaluates individual risk factors specifically for heart disease.
* **Advanced AI Modeling**: Utilizes sophisticated AI algorithms to analyze health data and provide accurate heart disease risk assessments.
* **Actionable Recommendations**: Offers clear and practical advice based on assessment results, such as lifestyle changes and medical follow-ups.
* **User-Friendly Interface**: Easy-to-use interface for inputting health data and viewing heart disease risk assessment results.

## **How It Works:**

* **User Input**: Users input their health information, such as age, BMI, physical activity levels, smoking status, medical history (e.g., previous heart attacks, strokes, diabetes), etc.
* **Data Analysis**: The app analyzes the input data using advanced AI models tailored for heart disease risk prediction.
* **Risk Assessment**: Provides a personalized risk score for heart disease.
* **Recommendations**: Offers actionable recommendations to mitigate the risk of heart disease, including lifestyle modifications.

## **Dataset:**

* The Behavioral Risk Factor Surveillance System (BRFSS) is the nation’s premier system of health-related telephone surveys that collect state data about U.S. residents regarding their health-related risk behaviors, chronic health conditions, and use of preventive services. Established in 1984 with 15 states, BRFSS now collects data in all 50 states as well as the District of Columbia and three U.S. territories. CDC BRFSS completes more than 400,000 adult interviews each year, making it the largest continuously conducted health survey system in the world.
* The dataset was sourced from Kaggle [(Behavioral Risk Factor Surveillance System (BRFSS) 2022)](https://www.kaggle.com/datasets/ariaxiong/behavioral-risk-factor-surveillance-system-2022/data) and it was originally downloaded from the [CDC BRFSS 2022 website.](https://www.cdc.gov/brfss/annual_data/annual_2022.html)
* To get more understanding regarding the dataset, please go to the [data_directory](./data_directory) folder.

## **Future Work:**

* **Enhanced Hyperparameter Tuning:** While initial hyperparameter tuning has been conducted, further experiments and refinements are planned to identify the optimal settings for the best model. Additional time will be dedicated to exploring a wider range of hyperparameters and employing advanced tuning techniques to enhance model performance.

* **Classification Threshold Tuning:** To achieve a better balance between false positives and true positives, we will focus on tuning the classification thresholds. This will help in optimizing the trade-off between sensitivity and specificity, ensuring more accurate predictions.

* **Feature Selection Improvements:** Further efforts will be made to refine the feature selection process. By carefully analyzing and selecting the most relevant features, we aim to improve the overall performance of the model. This includes experimenting with different feature selection techniques to enhance the predictive power of our app.

## **Installation**

To set up and run the AI-powered heart disease risk assessment app, follow these steps:

### Step 1: Clone the Repository:
```sh
git clone https://github.com/akthammomani/AI_powered_heart_disease_risk_assessment_app.git
cd AI_powered_heart_disease_risk_assessment_app
```
### Step 2: Set Up a Virtual Environment

Create and activate a virtual environment to manage dependencies:

For Windows:
```sh
python -m venv venv
venv\Scripts\activate
```
For macOS and Linux:
```sh
python3 -m venv venv
source venv/bin/activate
```
### Step 3: Install Dependencies

Install the required Python packages using pip:
```sh
pip install -r requirements.txt
```

### Step 4: Run the Application

To start the Streamlit application, run the following command:
```sh
streamlit run heart_app.py
```
## **License**

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

## **Acknowledgments**

Thank you to Professor Dallin for your guidance and support throughout this project/class. Your insights have been greatly appreciated.


  










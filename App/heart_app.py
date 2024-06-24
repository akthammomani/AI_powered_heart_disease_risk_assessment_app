import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
from PIL import Image
import io
from lightgbm import LGBMClassifier
import category_encoders as ce 
from imblearn.ensemble import EasyEnsembleClassifier


# Load the pickled model and encoder
with open('best_model.pkl', 'rb') as model_file:
    model = pkl.load(model_file)

with open('cbe_encoder.pkl', 'rb') as encoder_file:
    encoder = pkl.load(encoder_file)

# Load the dataset for reference
#data = pd.read_csv('brfss2022_data_wrangling_output.csv')
data = pd.read_csv('brfss2022_data_wrangling_output.zip', compression='zip')

# Convert the target variable to numerical values
data['heart_disease'] = data['heart_disease'].apply(lambda x: 1 if x == 'yes' else 0).astype('int')

icon = Image.open("heart_disease.JPG")

# Now, let's change page config by changing layout to 'Wide, give the app a new name and then change the icon:
st.set_page_config(layout='wide', page_title='AI-Powered Heart Disease Assessment', page_icon=icon)


# Set the title of the app
#st.title("AI-Powered Heart Disease Assessment")

# Main layout with three columns

row0_0, row0_1, row0_2, row0_3= st.columns((0.08,6,3,0.17))

with row0_1:
    st.title("AI-Powered Heart Disease Assessment App")
    st.write(
    """
    Unmatched Accuracy with Cutting-Edge Machine Learning Models
    """
    )
st.write('---')  

row0_a0, row0_b1, row0_d3= st.columns((0.08,6,0.17))
with row0_b1:
    st.write(
    """
      ## **Introduction**
    """)

with row0_b1:
    st.write(
    """
    The AI-Powered Heart Disease Risk Assessment App provides users with tailored risk scores and actionable recommendations to help mitigate their heart disease risk. Using advanced AI and modeling techniques, 
    this app offers easy-to-understand assessments and preventive measures to make safeguarding your cardiovascular health straightforward and accessible. 

    """)
st.write('---')

row1_a0, row2_b1, row3_d3= st.columns((0.08,6,0.17))
with row2_b1:
    st.write(
    """
      ## **How It Works:**
    """)

with row2_b1:
    st.write(
    """
    * **User Input:** Enter your health information, such as age, BMI, physical activity levels, smoking status, and medical history (e.g., heart attacks, strokes, diabetes).
    * **Data Analysis:** The app analyzes your input using advanced AI models specifically designed for heart disease risk prediction.
    * **Risk Assessment:** Receive a personalized risk score indicating your potential for heart disease.
    * **Recommendations:** Get actionable advice to mitigate your risk, including lifestyle modification suggestions.

    """)

#row0_3.image(Image.open('USD_logo.JPG'), use_column_width=True, width=3)

row1_0, row1_1, row1_2, row1_3, row1_5= st.columns((0.08,3,3,3,0.17))
with row1_1:
    st.write(
    """
    #### **Demographics**
    """
    )
row2_0, row2_1, row2_2, row2_3, row2_5= st.columns((0.08,3,3,3,0.17))


gender = row2_1.selectbox("What is your gender?", ["female", "male", "nonbinary"], index=1)
race = row2_2.selectbox("What is your race/ethnicity?", [
    "white_only_non_hispanic", "black_only_non_hispanic", "asian_only_non_hispanic", 
    "american_indian_or_alaskan_native_only_non_hispanic", "multiracial_non_hispanic", 
    "hispanic", "native_hawaiian_or_other_pacific_islander_only_non_hispanic"
], index=0)
age_category = row2_3.selectbox("What is your age group?", [
        "Age_18_to_24", "Age_25_to_29", "Age_30_to_34","Age_35_to_39", 
        "Age_40_to_44", "Age_45_to_49", "Age_50_to_54", "Age_55_to_59",
        "Age_60_to_64", "Age_65_to_69","Age_70_to_74","Age_75_to_79",
        "Age_80_or_older"
               ], index=12)

row3_0, row3_1, row3_2, row3_3, row3_5= st.columns((0.08,3,3,3,0.17))
with row3_1:
    st.write(
    """
    #### **Medical History**
    """
    )
row4_0, row4_1, row4_2, row4_3, row4_5= st.columns((0.08,3,3,3,0.17))

# row4_1
general_health = row4_1.selectbox("How would you rate your overall health?", ["excellent", "very_good", "good", "fair", "poor"], index=4)
heart_attack = row4_1.selectbox("Have you ever been diagnosed with a heart attack?", ["yes", "no"], help="A heart attack occurs when blood flow to part of the heart is blocked!", index=0)
kidney_disease = row4_1.selectbox("Has a doctor ever told you that you have kidney disease?", ["yes", "no"], index=0)
asthma = row4_1.selectbox("Have you ever been diagnosed with asthma?", ["never_asthma", "current_asthma", "former_asthma"], index=1)
could_not_afford_to_see_doctor = row4_1.selectbox("Have you ever been unable to see a doctor when needed due to cost?", ["yes", "no"], index=0)     
#row4_2
health_care_provider = row4_2.selectbox("Do you have a primary health care provider?", ["yes_only_one", "more_than_one", "no"], index=2)
stroke = row4_2.selectbox("Have you ever been diagnosed with a stroke?", ["yes", "no"], help="A stroke happens when blood supply to part of the brain is interrupted!", index=0)
diabetes = row4_2.selectbox("Have you ever been diagnosed with diabetes?", ["yes", "no", "no_prediabetes", "yes_during_pregnancy"], key="diabetes", index=0)
bmi = row4_2.selectbox("What is your body mass index (BMI)?", [
        "underweight_bmi_less_than_18_5", "normal_weight_bmi_18_5_to_24_9", "overweight_bmi_25_to_29_9",  
        "obese_bmi_30_or_more"
        
], help="BMI is a measure of body fat based on height and weight. Please use the BMI calculator at https://www.nhlbi.nih.gov/health/educational/lose_wt/BMI/bmicalc.htm", index=3)
length_of_time_since_last_routine_checkup = row4_2.selectbox("How long has it been since your last routine checkup?", ["past_year", "past_2_years", "past_5_years", "5+_years_ago", "never"], index=4)
#row4_3
#checkup_routine = row4_3.selectbox("Length of Time Since Last Routine Checkup", ["past_year", "never", "past_2_years", "past_5_years", "5+_years_ago"], key="checkup_routine")
depressive_disorder = row4_3.selectbox("Has a doctor ever told you that you have a depressive disorder?", ["yes", "no"], help="A depressive disorder is a medical condition characterized by persistent feelings of sadness, loss of interest, and other emotional and physical symptoms!", index=0)
physical_health = row4_3.selectbox("How many days in the past 30 days was your physical health not good?", ["zero_days_not_good", "1_to_13_days_not_good", "14_plus_days_not_good"], key="physical_health", index=2)
mental_health = row4_3.selectbox("How many days in the past 30 days was your mental health not good?", ["zero_days_not_good", "1_to_13_days_not_good", "14_plus_days_not_good"], key="mental_health", index=2)
walking = row4_3.selectbox("Do you have difficulty walking or climbing stairs?", ["yes", "no"], key="walking", index=0)

row5_0, row5_1, row5_2, row5_3, row5_5= st.columns((0.08,3,3,3,0.17))
with row5_1:
    st.write(
    """
    #### **Lifestyle**
    """
    )

row6_0, row6_1, row6_2, row6_3, row6_5= st.columns((0.08,3,3,3,0.17))
smoking_status = row6_1.selectbox("What is your smoking status?", [
    "never_smoked", "former_smoker", "current_smoker_some_days", "current_smoker_every_day"], key="smoking_status", index=3)
sleep_category = row6_1.selectbox("How many hours of sleep do you get on a typical night?", [
    "very_short_sleep_0_to_3_hours", "short_sleep_4_to_5_hours", "normal_sleep_6_to_8_hours",  
    "long_sleep_9_to_10_hours", "very_long_sleep_11_or_more_hours"], key="sleep_category", index=0)
drinks_category = row6_2.selectbox("How many alcoholic drinks do you consume in a typical week?", [
    "did_not_drink", "very_low_consumption_0.01_to_1_drinks", "low_consumption_1.01_to_5_drinks",  
    "moderate_consumption_5.01_to_10_drinks", "high_consumption_10.01_to_20_drinks", "very_high_consumption_more_than_20_drinks"], key="drinks_category", index=5)
binge_drinking_status = row6_2.selectbox("Have you engaged in binge drinking in the past 30 days?", ["yes", "no"], help="Binge drinking is consuming 5 or more drinks for men, or 4 or more drinks for women, in about 2 hours!", index=0)
exercise_status = row6_3.selectbox("Have you exercised in the past 30 days?", ["yes", "no"], key="exercise_status", index=0)
with row6_1:
    st.write(
    """
    #### **Learn More**
    [![](https://img.shields.io/badge/GitHub%20-Features%20Information-informational)](https://github.com/akthammomani/AI_powered_heart_disease_risk_assessment_app/tree/main/Notebooks/Exploratory_Data_Analysis/)
    """)

# Collect input data
input_data = {
    'gender': gender,
    'race': race,
    'general_health': general_health,
    'health_care_provider': health_care_provider,
    'could_not_afford_to_see_doctor': could_not_afford_to_see_doctor,
    'length_of_time_since_last_routine_checkup': length_of_time_since_last_routine_checkup,
    'ever_diagnosed_with_heart_attack': heart_attack,
    'ever_diagnosed_with_a_stroke': stroke,
    'ever_told_you_had_a_depressive_disorder': depressive_disorder,
    'ever_told_you_have_kidney_disease': kidney_disease,
    'ever_told_you_had_diabetes': diabetes,
    'BMI': bmi,
    'difficulty_walking_or_climbing_stairs': walking,
    'physical_health_status': physical_health,
    'mental_health_status': mental_health,
    'asthma_Status': asthma,
    'smoking_status': smoking_status,
    'binge_drinking_status': binge_drinking_status,
    'exercise_status_in_past_30_Days': exercise_status,
    'age_category': age_category,
    'sleep_category': sleep_category,
    'drinks_category': drinks_category
}

# Define a function to make predictions
def predict_heart_disease_risk(input_data, model, encoder):
    input_df = pd.DataFrame([input_data])
    input_encoded = encoder.transform(input_df, y=None, override_return_df=False)
    prediction = model.predict_proba(input_encoded)[:, 1][0] * 100
    return prediction

st.write('---')   
row8_0, row8_1, row8_5= st.columns((0.08,12,0.17))

with row8_1:
    st.write(
    """
    #### **AI Heart Disease Risk Assessment**
    """
    )

# This button will control the heart of the APP which is the Machine Learning part :)
btn1 = row8_1.button('Get Your Heart disease Risk Assessment')

if btn1:
    try:

        risk = predict_heart_disease_risk(input_data, model, encoder)
        with row8_1: 
            st.write(f"Predicted Heart Disease Risk: {risk:.2f}%")

            # Provide recommendations based on risk and user inputs
            recommendations = []

            if risk > 75:
                recommendations.append("Your risk of heart disease is very high. Here are some recommendations to reduce your risk:")
            elif risk > 50:
                recommendations.append("Your risk of heart disease is high. Here are some recommendations to reduce your risk:")
            elif risk > 25:
                recommendations.append("Your risk of heart disease is moderate. Here are some recommendations to reduce your risk:")
            else:
                recommendations.append("Your risk of heart disease is low. Keep up the good work and continue to maintain a healthy lifestyle.")

            if heart_attack == "yes":
                recommendations.append("- Regularly visit your cardiologist and adhere to prescribed medications.")
                recommendations.append("- Monitor any new or worsening symptoms and seek immediate medical attention if needed.")       
            if stroke == "yes":
                recommendations.append("- Follow your neurologist's recommendations and take prescribed medications consistently.")
                recommendations.append("- Engage in approved physical therapy or exercises to regain strength and mobility.")
            if general_health in ["fair", "poor"]:
                recommendations.append("- Focus on improving your overall health through a balanced diet and regular check-ups.")
            if bmi in ["overweight_bmi_25_to_29_9", "obese_bmi_30_or_more"]:
                recommendations.append("- Maintain a healthy weight through diet and exercise.")
            if smoking_status != "never_smoked":
                recommendations.append("- Quit smoking to significantly reduce your risk of heart disease.")
            if exercise_status == "no":
                recommendations.append("- Engage in regular physical activity to improve your heart health.")       
            if binge_drinking_status == "yes" or drinks_category in ["high_consumption_10.01_to_20_drinks", "very_high_consumption_more_than_20_drinks"]:
                recommendations.append("- Limit alcohol consumption to lower your risk.")                                     
            if sleep_category in ["short_sleep_4_to_5_hours", "very_short_sleep_0_to_3_hours"]:
                recommendations.append("- Consider aiming for 7-9 hours of quality sleep each night. Adequate sleep is crucial for maintaining heart health.")
            for recommendation in recommendations:
                st.write(recommendation)
    except Exception as e:
        row8_1.error(e)

with row8_1:
    st.write(
    """
    #### **Learn More**
    [![](https://img.shields.io/badge/GitHub%20-Machine%20Learning%20Models-informational)](https://github.com/akthammomani/AI_powered_heart_disease_risk_assessment_app/tree/main/Notebooks/Modeling/)
    """)

with row8_1:
    st.write(
    """
    ###### ***Disclaimer***
    *This app is not a replacement for professional medical advice, diagnosis, or treatment. Always consult your doctor or a qualified healthcare provider with any questions you may have regarding your health.*
    """)    
st.write('---')   
null9_0,row9_1 ,row9_2= st.columns((0.02,5,0.05)) 
with row9_1.expander("Leave Us a Comment or Question"):
    contact_form = """
                    <form action=https://formsubmit.co/aktham.momani81@gmail.com method="POST">
                        <input type="hidden" name="_captcha" value="false">
                        <input type="text" name="name" placeholder="Your name" required>
                        <input type="email" name="email" placeholder="Your email" required>
                        <textarea name="message" placeholder="Your message here"></textarea>
                        <button type="submit">Send</button>
                    </form>
                    """
    st.markdown(contact_form, unsafe_allow_html=True)

    # Use Local CSS File
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)                

    local_css("style.css")


null10_0,row10_1,row10_2= st.columns((0.04,7, 0.4))  
with row10_1:
    st.write(
    """
    ### **Contacts**
    [![](https://img.shields.io/badge/GitHub-Follow-informational)](https://github.com/akthammomani)
    [![](https://img.shields.io/badge/Linkedin-Connect-informational)](https://www.linkedin.com/in/akthammomani/)
    [![](https://img.shields.io/badge/Open-Issue-informational)](https://github.com/akthammomani/AI_powered_heart_disease_risk_assessment_app/issues)
    [![MAIL Badge](https://img.shields.io/badge/-aktham.momani81@gmail.com-c14438?style=flat-square&logo=Gmail&logoColor=white&link=mailto:aktham.momani81@gmail.com)](mailto:aktham.momani81@gmail.com)

    ###### © Aktham Momani, 2024
    """
)    


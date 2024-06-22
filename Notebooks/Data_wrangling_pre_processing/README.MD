# Data Wrangling and Pre-Processing: AI-Powered Heart Disease Risk Assessment

<center>
    <img src="https://github.com/akthammomani/AI_powered_heart_disease_risk_assessment_app/assets/67468718/2cab2215-ce7f-4951-a43a-02b88a5b9fa9" alt="wrnagling">
</center>

## Introduction

This notebook is dedicated to the crucial task of data wrangling, a foundational step in preparing our dataset for analysis and modeling. Data wrangling involves transforming and mapping raw data into a format suitable for analysis, ensuring data quality, consistency, and usability. The steps taken in this notebook include:

### Key Steps in Data Wrangling and Pre-Processing

- **Replacing Specific Values with NaN**: Standardized missing data representation by replacing erroneous or placeholder values with `NaN`.
- **Calculating Value Distribution**: Analyzed the distribution of existing values to understand the baseline state of the data.
- **Imputing Missing Values**: Used data distribution to impute missing values, maintaining the dataset's integrity.
- **Mapping Categorical Values**: Converted numeric codes into meaningful categorical labels for better interpretability.
- **Renaming Columns**: Updated column names to accurately reflect their contents, improving dataset readability and usability.

## Steps Taken

1. **Dealing with Missing Data**:
    - Identified missing values in the dataset.
    - Replaced specific values (e.g., 7 and 9) with `NaN` to standardize missing data representation.
    - Calculated the distribution of existing values to understand the data's baseline state.
    - Imputed missing values based on the distribution of existing values to ensure the data remains representative of its original characteristics.

2. **Data Mapping**:
    - Created a mapping dictionary to convert numeric codes into meaningful categorical labels (e.g., gender mapping: {1: 'male', 2: 'female', 3: 'nonbinary'}).
    - Applied the mapping to the relevant column.

3. **Renaming Columns**:
    - Updated column names to accurately reflect their contents, enhancing the clarity and usability of the dataset.

4. **Feature Engineering**:
    - Generated new features from existing ones to improve the dataset's predictive power and analytical value.


## Conclusion

This notebook ensures that the dataset is clean, consistent, and ready for subsequent analysis and machine learning tasks. By systematically applying the steps outlined above, we enhance the dataset's quality, making it more suitable for building reliable and robust predictive models.




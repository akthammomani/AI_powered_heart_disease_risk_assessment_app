# Extracting Descriptive Column Names for the Dataset

The Behavioral Risk Factor Surveillance System (BRFSS) dataset available on Kaggle, found [here](https://www.cdc.gov/brfss/annual_data/annual_2022.html), contains a wealth of information collected through surveys. However, the column names in the dataset are represented by short labels or codes (e.g., _STATE, FMONTH, IDATE), which can be difficult to interpret without additional context.

To ensure we fully understand what each column in the dataset represents, it is crucial to replace these short codes with their corresponding descriptive names. These descriptive names provide clear insights into the type of data each column holds, making the dataset easier to understand and analyze.

**Process Overview:**
* **Identify the Source for Descriptive Names:** The descriptive names corresponding to these short labels are typically documented in the [codebook in HTML](https://github.com/akthammomani/AI_powered_health_risk_assessment_app/tree/main/data_directory) or metadata provided by the data collection authority. In this case, the descriptive names are found in an HTML document provided by the BRFSS.

* **Parse the HTML Document:** Using web scraping techniques, such as BeautifulSoup in Python, we can parse the HTML document to extract the relevant information. Specifically, we look for tables or sections in the HTML that list the short labels alongside their descriptive names.

* **Match and Replace:** We create a mapping of short labels to their descriptive names. This mapping is then applied to our dataset to replace the short labels with more meaningful descriptive names.

* **Save the Enhanced Dataset:** The dataset with descriptive column names is saved for subsequent analysis, ensuring that all users can easily interpret the columns.

```python
#Let's import the necessary packages:
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

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

```


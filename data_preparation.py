# Import necessary libraries
import pandas as pd
import os

# Read the data from 'IMO_Team_2015.csv' file and select only the 'name', '_code', and 'total' columns
imo2015_tasks_points = pd.read_csv('data/IMO_Team_2015.csv')[['name', '_code', 'total']]

# Rename the columns to make them more understandable and use 'Country Code' as a common key for merging in the future
imo2015_tasks_points.rename(columns={'_code': 'Country Code', 'total': 'imo2015_tasks_points', 'name': 'Country Name'}, inplace=True)

# Create a list of all the files in the 'data/' directory except for 'IMO_Team_2015.csv'
files = [file for file in os.listdir('data/') if file != 'IMO_Team_2015.csv']

# Create a new DataFrame 'Mydata' with the data from 'imo2015_tasks_points' as starting point
Mydata = pd.DataFrame(imo2015_tasks_points)

# Loop through each file in the 'files' list and merge its data with 'Mydata' based on the 'Country Code'
for file in files:
    # Read data from the current file, selecting only 'Country Code' and '2015' columns
    df = pd.read_csv('data/' + file)[['Country Code', '2015']]

    # Rename the '2015' column to the current file's name without the '.csv' extension
    df.rename(columns={'2015': file[:-4]}, inplace=True)

    # Merge the data from the current file with 'Mydata' using the 'Country Code' as the common key
    Mydata = pd.merge(Mydata, df, on='Country Code')

# Drop rows with missing values (NaN) from the DataFrame 'Mydata'
Mydata.dropna(inplace=True)

# Reset the index of 'Mydata' after dropping rows
Mydata.reset_index(drop=True, inplace=True)

# Get the list of column names in 'Mydata' that are numeric, excluding 'name' and 'Country Code'
numeric_cols = Mydata.columns.drop(['Country Name', 'Country Code'])

# Replace any commas in the numeric columns with periods (decimal points) and convert them to float data type
Mydata[numeric_cols] = Mydata[numeric_cols].replace(',', '.', regex=True).astype(float)

# Divide specific columns by 100 to convert them from percentages to decimal values
Mydata[['gov_educ_expenditure', 'gross_enrollment_ratio', 'internet_user_percentage', 'prob_of_death',
        'unemployment_rate']] = Mydata[['gov_educ_expenditure', 'gross_enrollment_ratio', 'internet_user_percentage',
                                        'prob_of_death', 'unemployment_rate']].div(100)

# Sort data to make it easier to read
Mydata=Mydata.sort_values('Country Code', ascending=True)

# Save the final processed DataFrame 'Mydata' into a new CSV file 'Mydata.csv' without the index column
Mydata.to_csv('Mydata.csv', index=False)

import pandas as pd
import matplotlib.pyplot as plt

# Read the Excel file
excel_file = '/home/stu15/MachineLearning_ageEstimation/Code/ptb-xl-a-large/ptbxl_database_new.csv'
df = pd.read_csv(excel_file)

# Extract the age column
age_column = df['age']

# Remove missing values if any
age_column = age_column.dropna()
print("Number of entries ",len(age_column))
# Plot the histogram of ages
plt.hist(age_column, bins=20, edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Histogram of Age in ptb-xl dataset')
plt.grid(axis='y')

# Save the histogram as an image
output_file = '/home/stu15/MachineLearning_ageEstimation/Code/JAINIK/domain_data_exploring/ptb-xl.png'
plt.savefig(output_file)
plt.show()

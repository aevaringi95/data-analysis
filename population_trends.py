'''
This codes takes in data from Hagstofan, transforms it and 
creates a times series plot
'''



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data, skipping the first two rows
data = pd.read_excel("data/MAN00101_20230711-195038.xlsx", skiprows=2)

# Rename columns
data.columns = ["Group", "Age"] + list(data.columns[2:].astype(int))

# Melt the dataframe into long format suitable for time series analysis
data_melted = data.melt(id_vars=["Group", "Age"], var_name="Year", value_name="Population")

# Forward fill missing values in 'Group' column
data_melted['Group'] = data_melted['Group'].fillna(method='ffill')

# Filter data for 'Alls' (everyone), 'Karlar' (men), and 'Konur' (women)
data_filtered = data_melted[data_melted['Group'].isin(['Alls', 'Karlar', 'Konur']) & (data_melted['Age'] == 'Alls')]

# Set the style of the plot
sns.set(style="whitegrid")

# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Draw line plot of time series
sns.lineplot(x='Year', y='Population', hue='Group', data=data_filtered, ax=ax)

# Set title and labels
ax.set_title('Population of Iceland from 1841 to 2023', fontsize=16)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Population', fontsize=12)

# Show the plot
plt.show()

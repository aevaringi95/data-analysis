import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA

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

# Extract data for 'Alls' (total population)
total_population = data_filtered[data_filtered['Group'] == 'Alls']

# Calculate average growth rate for the last 10 years (2013-2023)
growth_rates = total_population['Population'].pct_change()
average_growth_rate = growth_rates.loc[total_population['Year'] >= 2013].mean()

# Initialize predictions
predictions_simple = total_population.copy()

# Predict population for 2024 to 2050 using the simple method
for year in range(2024, 2051):
    last_population = predictions_simple['Population'].iloc[-1]
    next_population = last_population * (1 + average_growth_rate)
    new_row = pd.DataFrame({
        'Year': [year],
        'Population': [next_population],
        'Group': ['Alls (Simple Forecast)']
    }, columns=['Year', 'Population', 'Group'])
    predictions_simple = pd.concat([predictions_simple, new_row], ignore_index=True)

# Fit an AutoReg model to the data
model_ar = AutoReg(total_population['Population'], lags=1)
model_fit_ar = model_ar.fit()

# Predict population for 2024 to 2050 using the AR model
start = len(total_population)
end = start + (2050 - 2023)
predictions_ar = model_fit_ar.predict(start=start, end=end)

# Exclude the in-sample predictions from the AR forecasts
num_forecasts = 2050 - 2023 + 1  # +1 to include 2050
predictions_ar_out_of_sample = predictions_ar[-num_forecasts:]

# Create a DataFrame for the AR predictions
predictions_ar_df = pd.DataFrame({
    'Year': range(2023, 2051),  # Adjust to match the number of out-of-sample predictions
    'Population': predictions_ar_out_of_sample.values,
    'Group': 'Alls (AR Forecast)'
})

# Fit an ARIMA model to the data
model_arima = ARIMA(total_population['Population'], order=(2, 2, 2))
model_fit_arima = model_arima.fit()

# Predict population for 2024 to 2050 using the ARIMA model
predictions_arima = model_fit_arima.predict(start=start, end=end, typ='levels')

# Exclude the in-sample predictions from the ARIMA forecasts
predictions_arima_out_of_sample = predictions_arima[-num_forecasts:]

# Create a DataFrame for the ARIMA predictions
predictions_arima_df = pd.DataFrame({
    'Year': range(2023, 2051),  # Adjust to match the number of out-of-sample predictions
    'Population': predictions_arima_out_of_sample.values,
    'Group': 'Alls (ARIMA Forecast)'
})

# Concatenate the actual data and all the predictions
total_and_all_forecasts = pd.concat([total_population, predictions_simple, predictions_ar_df, predictions_arima_df])

# Create a new figure and axis for the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the actual data
sns.lineplot(x='Year', y='Population', data=total_population, ax=ax, label='Actual Data')

# Plot the simple forecast
sns.lineplot(x='Year', y='Population', data=predictions_simple[predictions_simple['Year'] > 2023], ax=ax, label='Simple Forecast', linestyle='--')

# Plot the AR forecast
sns.lineplot(x='Year', y='Population', data=predictions_ar_df, ax=ax, label='AR Forecast', linestyle=':')

# Plot the ARIMA forecast
sns.lineplot(x='Year', y='Population', data=predictions_arima_df, ax=ax, label='ARIMA Forecast', linestyle='-.')

# Set title and labels
ax.set_title('Population of Iceland from 1841 to 2050 (Forecasts)', fontsize=16)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Population', fontsize=12)

# Show the legend
ax.legend()

# Show the plot
plt.show()

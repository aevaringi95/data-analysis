import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# Define the p, d, and q parameters to take any value between 0 and 2
p = d = q = range(0, 3)

# Generate all different combinations of p, d, and q triplets
pdq = list(itertools.product(p, d, q))

# Initialize the best AIC to a large value
best_aic = np.inf
best_pdq = None

# Grid search over p, d, q parameters
for combo in pdq:
    try:
        # Fit the ARIMA model
        model_arima = ARIMA(total_population['Population'], order=combo)
        model_fit_arima = model_arima.fit()

        # If the current run of ARIMA provides a better AIC than the best one so far, update the best AIC and best pdq
        if model_fit_arima.aic < best_aic:
            best_aic = model_fit_arima.aic
            best_pdq = combo
    except:
        continue

print('Best ARIMA parameters:', best_pdq)

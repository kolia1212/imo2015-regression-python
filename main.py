import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import statsmodels.stats.diagnostic
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.api import linear_rainbow
from statsmodels.stats.diagnostic import linear_reset
from statsmodels.stats.outliers_influence import OLSInfluence

# Load the data
data_path = "C:/Users/User/PycharmProjects/WORK/imo_python/Mydata.csv"
Mydata = pd.read_csv(data_path)

# Define the independent and dependent variables
X = Mydata.drop(columns=['imo2015_tasks_points', 'Country Name', 'Country Code'])
y = Mydata['imo2015_tasks_points']

# Fit the initial linear regression model
X = sm.add_constant(X)  # Add a constant for the intercept
model = sm.OLS(y, X).fit()

# Summary of the initial model
print("Initial Model Summary:")
print(model.summary())
print()

# Check assumptions for the initial model

# Multicollinearity check - Calculate the correlation matrix and variance inflation factors (VIF)
vif = pd.DataFrame()
vif["Features"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

corr = X.drop(columns=['const']).corr()
print("First model: Multicollinearity Check - Correlation Matrix:")
print(corr)
print()

print("First model: Multicollinearity Check - Variance Inflation Factors:")
print(vif)
print()

# Normality of residuals check

# Calculate the residuals and create plots to check for normality
residuals = model.resid
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
fig.canvas.manager.set_window_title('Model 1 - Assumption Checks')

# Histogram of residuals
axs[0, 0].hist(residuals, density=True)
axs[0, 0].set_title('Histogram of Residuals')
sns.kdeplot(residuals, color='grey', ax=axs[0, 0])

# Q-Q plot
stats.probplot(residuals, plot=axs[0, 1])
axs[0, 1].set_title('Normal Q-Q Plot')

# Anderson-Darling Test for Normality
anderson_result = stats.anderson(residuals, dist='norm')
p_value = anderson_result.significance_level[np.searchsorted(anderson_result.critical_values, anderson_result.statistic)] / 100
print("First model: Normality of Residuals - Anderson-Darling Test:")
print("P-value:", p_value)
# The p-value could be larger in reality. However, the anderson_result.significance_level does not have values greater than 15%.
# Nevertheless, in this situation, we do not reject the null hypothesis about the normality of residuals
print()

# Shapiro-Wilk Test for Normality
_, p_value = stats.shapiro(residuals)
print("First model: Normality of Residuals - Shapiro-Wilk Test:")
print("P-value:", p_value)
print()

# Homoscedasticity check - Breusch-Pagan Test
_, p_bp, _, _ = het_breuschpagan(model.resid, X)
print("First model: Homoscedasticity - Breusch-Pagan Test:")
print("P-value:", p_bp)
print()

# Fitted Values vs Residuals Plot
axs[1, 0].scatter(model.fittedvalues, residuals, alpha=0.7)
axs[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)  # Add a horizontal line at y=0 for reference
axs[1, 0].set_xlabel('Fitted Values')
axs[1, 0].set_ylabel('Residuals')
axs[1, 0].set_title('Fitted Values vs. Residuals Plot')

# Autocorrelation check

# Durbin-Watson Test for Autocorrelation
print("First model: Autocorrelation - Durbin-Watson Test:")
print("Durbin-Watson test statistic:", durbin_watson(residuals))
print()

# Autocorrelation Function (ACF) of Residuals
plot_acf(residuals, ax=axs[1, 1])
axs[1, 1].set_title("Autocorrelation Function (ACF) of Residuals")

# Ljung-Box Test for Autocorrelation
p_value = acorr_ljungbox(residuals, lags=[1])['lb_pvalue'][1]
print("First model: Autocorrelation - Ljung-Box Test:")
print("P-value:", p_value)
print()

# Linearity check

# Rainbow Test
_, rainbow_p_value = linear_rainbow(model)
print("First model: Linearity - Rainbow Test:")
print("P-value:", rainbow_p_value)
print()

# RESET Test
res_np = sm.OLS(model.model.endog, model.model.exog).fit()
p_value = linear_reset(res_np).pvalue
print("First model: Linearity - RESET Test:")
print("P-value:", p_value)
print()


# Second part of the project

# Influence statistics
influence = OLSInfluence(model)
leverage = influence.hat_matrix_diag
cooks_distance = influence.cooks_distance[0]

# Identify high leverage points using a threshold of mean + 2 * std
high_leverage = np.where(leverage > np.mean(leverage) + 2 * np.std(leverage))[0]

# Identify high Cook's distance points using a threshold of mean + 2 * std
high_cooks_distance = np.where(cooks_distance > np.mean(cooks_distance) + 2 * np.std(cooks_distance))[0]

print("Influence Statistics:")
print("High leverage points:", high_leverage)#So that means, observations 27, 32, 46 i 50 are high leverage
print("High Cook's distance points:", high_cooks_distance)#So, observations 27, 46 has high Cook's distance
print()

# Backward elimination and second model
while True:
    model2 = sm.OLS(y, X).fit()
    p_values = model2.pvalues[1:]
    max_p_value = p_values.max()

    if max_p_value > 0.06:#Why not 0.05? It turns out, that pvalue of internet_user_percentage is 0.51, so I decided to set a bit higher significance level
        feature_to_remove = p_values.idxmax()
        X = X.drop(columns=[feature_to_remove])
    else:
        break

print("Model 2 Summary:")
print(model2.summary())
print()

residuals = model2.resid
# Anderson-Darling Test for Normality
anderson_result = stats.anderson(residuals, dist='norm')
p_value = anderson_result.significance_level[np.searchsorted(anderson_result.critical_values, anderson_result.statistic)] / 100
print("Second model: Normality of Residuals - Anderson-Darling Test:")
print("P-value:", p_value)

# Shapiro-Wilk Test for Normality
_, p_value = stats.shapiro(residuals)
print("Second model: Normality of Residuals - Shapiro-Wilk Test:")
print("P-value:", p_value)
print()


# Third model
formula = 'imo2015_tasks_points ~ high_tech_exports + np.sqrt(area) + prob_of_death + np.exp(internet_user_percentage)'
model3 = sm.OLS.from_formula(formula, data=Mydata).fit()

print("Third Model Summary:")
print(model3.summary())
print()

X3 = X[['const', 'high_tech_exports', 'area', 'prob_of_death', 'internet_user_percentage']].copy()
X3['area'] = np.sqrt(X3['area'])
X3['internet_user_percentage'] = np.exp(X3['internet_user_percentage'])
X3.rename(columns={'area': 'sqrt(area)', 'internet_user_percentage': 'exp(internet_user_percentage)'}, inplace=True)

# Checking assumptions for the third model

# Multicollinearity
vif = pd.DataFrame()
vif["Features"] = X3.columns
vif["VIF"] = [variance_inflation_factor(X3.values, i) for i in range(len(X3.columns))]

print("Assumptions Check for the Third Model:")
print("Third model: Multicollinearity Check - Variance Inflation Factors:")
print(vif)
print()

# Normality of residuals
residuals = model3.resid
fig3, axs3 = plt.subplots(2, 2, figsize=(10, 8))
fig3.canvas.manager.set_window_title('Model 3 - Assumption Checks')

# Histogram of residuals
axs3[0, 0].hist(residuals, density=True)
sns.kdeplot(residuals, color='grey', ax=axs3[0, 0])
axs3[0, 0].set_title('Histogram of Residuals')

# Q-Q plot
stats.probplot(residuals, plot=axs3[0, 1])
axs3[0, 1].set_title('Normal Q-Q Plot')

# Anderson-Darling Test for Normality
anderson_result = stats.anderson(residuals, dist='norm')
p_value = anderson_result.significance_level[np.searchsorted(anderson_result.critical_values, anderson_result.statistic)] / 100
print("Third model: Normality of Residuals - Anderson-Darling Test:")
print("P-value:", p_value)

# Shapiro-Wilk Test for Normality
_, p_value = stats.shapiro(residuals)
print("Third model: Normality of Residuals - Shapiro-Wilk Test:")
print("P-value:", p_value)
print()

# Homoscedasticity

# Breusch-Pagan Test
_, p_value, _, _ = het_breuschpagan(model3.resid, X)
print("Third model: Homoscedasticity - Breusch-Pagan Test:")
print("P-value:", p_value)
print()

# Fitted Values vs Residuals Plot
axs3[1, 0].scatter(model3.fittedvalues, residuals, alpha=0.7)
axs3[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)  # Add a horizontal line at y=0 for reference
axs3[1, 0].set_xlabel('Fitted Values')
axs3[1, 0].set_ylabel('Residuals')
axs3[1, 0].set_title('Fitted Values vs. Residuals Plot')

# Autocorrelation

# Durbin-Watson Test for Autocorrelation
print("Third model: Autocorrelation - Durbin-Watson Test:")
print("Durbin-Watson test statistic:", durbin_watson(residuals))
print()

# Autocorrelation Function (ACF) of Residuals
plot_acf(residuals, ax=axs3[1, 1])
axs3[1, 1].set_title("Autocorrelation Function (ACF) of Residuals")

# Ljung-Box Test for Autocorrelation
p_value = acorr_ljungbox(residuals, lags=[1])['lb_pvalue'][1]
print("Third model: Autocorrelation - Ljung-Box Test:")
print("P-value:", p_value)
print()

# Linearity
# Rainbow Test
_, p_value = linear_rainbow(model3)
print("Third model: Linearity - Rainbow Test:")
print("P-value:", p_value)
print()

# RESET Test
res_np = sm.OLS(model3.model.endog, model3.model.exog).fit()
p_value = linear_reset(res_np).pvalue
print("Third model: Linearity - RESET Test:")
print("P-value:", p_value)
print()

# Model comparison

print("R-squared of 3 models:", model.rsquared, model2.rsquared, model3.rsquared)
print("Adjusted R-squared of 3 models:", model.rsquared_adj, model2.rsquared_adj, model3.rsquared_adj)
print("AIC of 3 models:", model.aic, model2.aic, model3.aic)
print("BIC of 3 models:", model.bic, model2.bic, model3.bic)
print()

# Show the plots
plt.tight_layout()
plt.show()

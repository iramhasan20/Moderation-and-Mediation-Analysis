import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from linearmodels.panel import RandomEffects
from scipy.stats import norm

# Load the dataset
df = pd.read_csv('data.csv')

# Rename variables for differentiation
CSR_VAR = 'X1'  # SCVPS
DV_VARS = ['Y1', 'Y2', 'Y3', 'Y4']  # ROA, ROE, TQ, PE
CONTROL_VARS = ['C1', 'C2', 'C3']  # MCAP, RISK, AGE
MODERATOR_VARS = ['M1', 'M2', 'M3', 'M4']  # FAM, GOV, INST, FORG
MEDIATOR_VAR = 'Z1'  # IC

# Create interaction terms
for mod in MODERATOR_VARS:
    df[f'{CSR_VAR}_{mod}'] = df[CSR_VAR] * df[mod]

# Define independent variables including interaction terms
independent_vars = [CSR_VAR] + MODERATOR_VARS + [f'{CSR_VAR}_{mod}' for mod in MODERATOR_VARS] + CONTROL_VARS

# Function to calculate VIF
def calculate_vif(df, independent_vars):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = independent_vars
    vif_data["VIF"] = [variance_inflation_factor(df[independent_vars].values, i) for i in range(len(independent_vars))]
    return vif_data

# Compute VIF values
vif_values = calculate_vif(df, independent_vars)
vif_values.to_excel('vif_values.xlsx', index=False)
print("VIF values saved.")

# Summary Statistics for Moderation and Mediation Analyses
print("Summary Statistics:")
print(df.describe())

# Correlation Matrix for Moderation Analysis
print("Correlation Matrix for Moderation Analysis:")
print(df[independent_vars].corr())

# Moderation Analysis using Random Effects Regression
def run_random_effects_regression(df, dependent_var, independent_vars, entity_var='Firm', time_var='Year'):
    df_panel = df.set_index([entity_var, time_var])
    model = RandomEffects(df_panel[dependent_var], df_panel[independent_vars])
    results = model.fit()
    return results

results_list = {}
for dep_var in DV_VARS:
    results = run_random_effects_regression(df, dep_var, independent_vars)
    results_list[dep_var] = results

# Correlation Matrix for Mediation Analysis
mediation_vars = [CSR_VAR, MEDIATOR_VAR] + CONTROL_VARS
print("Correlation Matrix for Mediation Analysis:")
print(df[mediation_vars].corr())

# Mediation Analysis using Baron and Kenny Approach
def extract_results(results):
    return pd.DataFrame({'coef': results.params, 'std_err': results.std_errors, 't': results.tstats, 'p_value': results.pvalues})

# Step 1: CSR affects CFP
model1 = RandomEffects.from_formula(f'Y1 ~ {CSR_VAR} + {" + ".join(CONTROL_VARS)}', data=df.set_index(['Firm', 'Year']))
results1 = model1.fit()

# Step 2: CSR affects Mediator
model2 = RandomEffects.from_formula(f'{MEDIATOR_VAR} ~ {CSR_VAR} + {" + ".join(CONTROL_VARS)}', data=df.set_index(['Firm', 'Year']))
results2 = model2.fit()

# Step 3: Mediator affects CFP, controlling for CSR
model3 = RandomEffects.from_formula(f'Y1 ~ {MEDIATOR_VAR} + {" + ".join(CONTROL_VARS)}', data=df.set_index(['Firm', 'Year']))
results3 = model3.fit()

# Step 4: CSR affects CFP, including Mediator
model4 = RandomEffects.from_formula(f'Y1 ~ {CSR_VAR} + {MEDIATOR_VAR} + {" + ".join(CONTROL_VARS)}', data=df.set_index(['Firm', 'Year']))
results4 = model4.fit()

# Robustness Check - Random Effects Panel Regression for Moderation
def run_random_effects_panel(df, dependent_var, independent_vars, entity_var='Firm', time_var='Year'):
    df_panel = df.set_index([entity_var, time_var])
    model = RandomEffects(df_panel[dependent_var], df_panel[independent_vars])
    results = model.fit()
    return results

random_effects_results_list = {}
for dep_var in DV_VARS:
    results = run_random_effects_panel(df, dep_var, independent_vars)
    random_effects_results_list[dep_var] = results

# Robustness Test -Sobel Test for Mediation
def sobel_test(a_coef, a_se, b_coef, b_se):
    sobel_stat = (a_coef * b_coef) / np.sqrt((b_coef**2 * a_se**2) + (a_coef**2 * b_se**2))
    p_value = 2 * (1 - norm.cdf(abs(sobel_stat)))
    return sobel_stat, p_value

# Extract coefficients and standard errors
a_coef, a_se = results2.params[CSR_VAR], results2.bse[CSR_VAR]
b_coef, b_se = results3.params[MEDIATOR_VAR], results3.bse[MEDIATOR_VAR]

# Compute Sobel test
test_stat, p_val = sobel_test(a_coef, a_se, b_coef, b_se)
print(f"Sobel Test Statistic: {test_stat}, P-Value: {p_val}")

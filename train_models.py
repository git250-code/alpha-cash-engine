import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error
import joblib
import os

# 1. Load data
print("Loading data...")
df = pd.read_csv('data/alphacash1.csv')

# 2. Sample 200k rows for memory
print("Sampling 200k rows...")
df_sample = df.sample(n=200_000, random_state=42)

# 3. Define features & targets
INPUT_COLS = [
    'current_bank_balance', 'monthly_expense', 'monthly_revenue',
    'runway_months', 'recurring_obligations', 'cash_inflows_next_30d',
    'sector', 'investment_style', 'startup_age_months',
    'burn_variability_index', 'compliance_flag', 'has_funding_round',
    'cash_utilization_rate'
]
TARGET_CLASS = 'suggested_action'
TARGET_REG = 'expected_return'

# Split features and labels
X = df_sample[INPUT_COLS]
y_class = df_sample[TARGET_CLASS]
y_reg = df_sample[TARGET_REG]

# 4. Train-test split
print("Splitting data...")
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42
)

# 5. Preprocessor
categorical = ['sector', 'investment_style', 'compliance_flag', 'has_funding_round']
numeric = [c for c in INPUT_COLS if c not in categorical]
preprocessor = ColumnTransformer([
    ('num', 'passthrough', numeric),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical)
])

# 6. Pipelines
clf_pipe = Pipeline([
    ('preproc', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=20, n_jobs=-1, random_state=42))
])
reg_pipe = Pipeline([
    ('preproc', preprocessor),
    ('reg', RandomForestRegressor(n_estimators=20, n_jobs=-1, random_state=42))
])

# 7. Train
print("Training classifier...")
clf_pipe.fit(X_train, y_class_train)
print("Training regressor...")
reg_pipe.fit(X_train, y_reg_train)

# 8. Evaluate
print("Evaluating classifier...")
print(classification_report(y_class_test, clf_pipe.predict(X_test), zero_division=0))
print("Evaluating regressor...")
rmse = mean_squared_error(y_reg_test, reg_pipe.predict(X_test)) ** 0.5
print(f"RMSE: {rmse:.2f}")

# 9. Save models
print("Saving models...")
os.makedirs('models', exist_ok=True)
joblib.dump(clf_pipe, 'models/alpha_cash_classifier_200k.joblib')
joblib.dump(reg_pipe, 'models/alpha_cash_regressor_200k.joblib')
print("Done.")

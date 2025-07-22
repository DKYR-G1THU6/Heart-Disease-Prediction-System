#Import Libraries for Pre-Processing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier #Random Forest for imputation
from sklearn.preprocessing import PowerTransformer #Data transformation
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm #Features Importance

path = '/content/HeartAssign2.csv'
df = pd.read_csv(path)

#---------------------------------
# EDA
#---------------------------------
# Count occurrences of each target value
target_counts = df['target'].value_counts().sort_index()

# Plotting the column chart
plt.figure(figsize=(8, 5))
plt.bar(target_counts.index.astype(str), target_counts.values, color='skyblue')
plt.xlabel('Target')
plt.ylabel('Count')
plt.title('Distribution of Heart Disease (Target Variable)')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

def eda_visualization(df):

    sns.set_style("whitegrid")

    categorical_features=['sex', 'cp', 'fbs', 'restecg', 'exang', 'ca', 'thal', 'slope']

    # Histogram for numerical features
    df.hist(figsize=(20, 20), bins=20, edgecolor='black', color='skyblue')
    plt.suptitle("Distribution of Numerical Features", fontsize=16)
    plt.show()

    #Bivariate graphs for categorical data
    for col in categorical_features:
        plt.figure(figsize=(8, 5))
        sns.countplot(x=df[col], hue=df["target"], palette="coolwarm")
        plt.title(f"{col} vs target")
        plt.show()

eda_visualization(df)

#---------------------------------
# Data Splitting
#---------------------------------
# 70% Training, 15% Validating, 15% Testing
train_data, temp_data = train_test_split(df, test_size=0.30, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

print("Train：", train_data.shape[0])
print("Valid：", val_data.shape[0])
print("Test：", test_data.shape[0])

#---------------------------------
# Data Preprocessing
#---------------------------------
train_data = train_data.drop_duplicates()
train_data= train_data.dropna(subset=['target'])
train_data = train_data[train_data['thal'] != 0]

# Function to impute missing values using Random Forest

def rf_impute_missing(data, col_name, train_data, is_categorical=False, round_to_int=False):
    df = data.copy()

    # Skip if no missing values
    if df[col_name].isna().sum() == 0:
        return df

    # Separate missing and non-missing values
    missing_rows = df[df[col_name].isna()]
    non_missing_rows = train_data[train_data[col_name].notna()]  # Use training data

    # Define predictors (exclude the column being imputed)
    X_train = non_missing_rows.drop(columns=[col_name]).fillna(train_data.median())
    y_train = non_missing_rows[col_name]

    X_missing = missing_rows.drop(columns=[col_name]).fillna(train_data.median())

    # Skip if no valid training data
    if X_train.shape[0] == 0 or X_missing.shape[0] == 0:
        return df

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42) if is_categorical else RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict missing values
    predicted_values = model.predict(X_missing)

    # If rounding is needed, apply rounding
    if round_to_int:
        predicted_values = np.round(predicted_values).astype(int)

    # Update DataFrame
    df.loc[df[col_name].isna(), col_name] = predicted_values

    return df

missing_values = train_data.isnull().sum()
total_missing = missing_values.sum()
print("Missing value：", total_missing)
print("Missing values:\n", train_data.isna().sum())

# Function to impute outlier values using a separate Random Forest model
def rf_impute_outliers(data, col_name, train_data, is_categorical=False, round_to_int=False):
    df = data.copy()

    # Identify outliers replaced with NaN
    outlier_rows = df[df[col_name].isna()]
    non_outlier_rows = train_data[train_data[col_name].notna()]  # Use training data

    # Define predictors
    X_train = non_outlier_rows.drop(columns=[col_name]).fillna(train_data.median())
    y_train = non_outlier_rows[col_name]

    X_outliers = outlier_rows.drop(columns=[col_name]).fillna(train_data.median())

    # Skip if no valid training data
    if X_train.shape[0] == 0 or X_outliers.shape[0] == 0:
        return df

    # Train separate model for outliers
    model = RandomForestClassifier(n_estimators=100, random_state=42) if is_categorical else RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict outlier values
    predicted_values = model.predict(X_outliers)

    # If rounding is needed, apply rounding
    if round_to_int:
        predicted_values = np.round(predicted_values).astype(int)

    # Update DataFrame
    df.loc[df[col_name].isna(), col_name] = predicted_values

    return df

# Function to detect outliers using 3 standard deviations and replace them with NaN
def replace_outliers_with_nan(df, features):
    df = df.copy()

    for feature in features:
      mean = df[feature].mean()
      std_dev = df[feature].std()

      lower_bound = mean - 3 * std_dev
      upper_bound = mean + 3 * std_dev
      df.loc[(df[feature] < lower_bound) | (df[feature] > upper_bound), feature] = np.nan

    return df

# Identify numerical features
numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Replace outliers with NaN
for i in range(2):
    train_data = replace_outliers_with_nan(train_data,numeric_features)

# Identify integer columns that should not be float
int_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'ca', 'thal', 'slope']

# Identify columns with missing values (including outliers replaced with NaN)
missing_cols_train = train_data.columns[train_data.isna().sum() > 0]

# Impute Missing Values
for col in missing_cols_train:
    if train_data[col].isna().sum() > 0:
        round_to_int = col in int_columns
        train_data = rf_impute_missing(train_data, col, train_data, False, round_to_int)

# Impute Outliers
for col in missing_cols_train:
    if train_data[col].isna().sum() > 0:
        round_to_int = col in int_columns
        train_data = rf_impute_outliers(train_data, col, train_data, False, round_to_int)

print("Missing values:\n", train_data.isna().sum())

# Remove duplicated data after imputation
train_data = train_data.drop_duplicates()

# Data transform
# Select the specific column to transform
column_to_transform = "oldpeak"

# Apply Yeo-Johnson transformation to only 'oldpeak'
pt = PowerTransformer(method='yeo-johnson', standardize=False)  # Disable standardization
train_data[column_to_transform] = pt.fit_transform(train_data[[column_to_transform]])

# Print transformed and recovered data for verification
print("Transformed 'oldpeak':\n", train_data[[column_to_transform]].head())

print("Skewness of oldpeak after power transform:\n", train_data.skew().sort_values(ascending=False))

# Standardization
scaler = StandardScaler()
columns_to_scale = ['trestbps','chol','thalach','oldpeak']

train_data[columns_to_scale] = scaler.fit_transform(train_data[columns_to_scale])

# Normalization
MinMax = MinMaxScaler()
New_age = ['age']

train_data[New_age] = MinMax.fit_transform(train_data[New_age])

# One-Hot Encoding
#One-Hot Encoding for columns with categorical value

def one_hot_encode_columns(df, columns, column_names_dict):

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    df_encoded = df.copy()

    for col in columns:
        encoded_array = encoder.fit_transform(df[[col]])  # One-hot encode
        default_feature_names = encoder.get_feature_names_out([col])  # Get feature names

        # Use custom names
        custom_names = column_names_dict.get(col, default_feature_names)
        if len(custom_names) != encoded_array.shape[1]:
            raise ValueError(f"Number of custom names for '{col}' does not match encoded shape")

        # Convert to DataFrame
        encoded_df = pd.DataFrame(encoded_array, columns=custom_names, index=df.index)

        # Drop the original column to prevent multicollinearity
        last_col = custom_names[1]  # second one-hot column to drop
        encoded_df.drop(columns=[last_col], inplace=True)

        df_encoded = df_encoded.drop(columns=[col]).join(encoded_df)

    return df_encoded

category_mapping = {
    'cp': ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"],
    'restecg': ["ECG Normal", "ECG Abnormal", "Left Ventricular Hypertrophy"],
    'thal': ["Thalassemia Normal", "Thalassemia Fixed defect", "Reversible defect"]
}

encoded_train_data = one_hot_encode_columns(train_data, ['cp', 'restecg', 'thal'], category_mapping)
print(encoded_train_data)

# save cleaned training dataset
encoded_train_data.to_csv('heart_train.csv',index=False)

# Compute the correlation matrix
corr_matrix = encoded_train_data.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))  # Set figure size
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

# Add title
plt.title("Correlation Heatmap")

# Show the plot
plt.show()

def find_high_correlation(df, threshold = 0.7):
    corr_matrix = df.corr();
    high_corr_pairs = [];

    for i in range (len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
          corr_value = corr_matrix.iloc[i,j]
          if (abs(corr_value) >= threshold):
            high_corr_pairs.append((corr_matrix.index[i], corr_matrix.columns[j], corr_value))

    return high_corr_pairs;

result = find_high_correlation(encoded_train_data,  0.7)
for pair in result:
  print(f"Correlation between {pair[0]} and {pair[1]}: {pair[2]:.2f}")

# Feature Importance using Random Forest
def plot_feature_importance(data):

    # Split into features and target
    X = data.drop(columns="target")
    y = data["target"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X, y)

    # Get feature importances
    importances = model.feature_importances_
    feature_names = X.columns

    # Create DataFrame for visualization
    feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

    # Plot feature importance
    plt.figure(figsize=(10, 5))
    plt.barh(feature_importance_df["Feature"], feature_importance_df["Importance"], color="skyblue")
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Features")
    plt.title("Feature Importance (Random Forest)")
    plt.gca().invert_yaxis()
    plt.show()

plot_feature_importance(encoded_train_data)

#---------------------------------
# Modelling
#---------------------------------
# Import Libraries for Modelling
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.experimental import enable_halving_search_cv  # Needed to enable HalvingGridSearchCV
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = "/content/heart_train.csv"
df = pd.read_csv(file_path)

# Extract features and target variable
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter = 500),
    "SVM": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier(criterion = 'entropy', max_depth = 7),
    "Random Forest": RandomForestClassifier(n_estimators = 500),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(DecisionTreeClassifier(min_samples_split=10,max_depth=4),n_estimators=10,learning_rate=0.6),
    "Extra Trees": ExtraTreesClassifier(),
    "Bagging": BaggingClassifier(),
    "XGBoost": XGBClassifier(eval_metric='logloss'),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(),
    "MLP (Neural Network)": MLPClassifier(max_iter=500),
    "LightGBM": LGBMClassifier(),
}

param_grids = {
    "Logistic Regression": {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs']
    },
    "SVM": {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    },
    "Decision Tree": {
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10]
    },
    "Random Forest": {
        'n_estimators': [100, 300, 500],
        'max_depth': [None, 10, 20]
    },
    "Gradient Boosting": {
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200]
    },
    "AdaBoost": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1]
    },
    "Extra Trees": {
        'n_estimators': [100, 300, 500],
        'max_depth': [None, 10, 20]
    },
    "Bagging": {
        'n_estimators': [10, 50, 100],
        'max_samples': [0.5, 1.0]
    },
    "XGBoost": {
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200],
        'max_depth': [3, 6, 9]
    },
    "KNN": {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance']
    },
    "Naive Bayes": {},  # no parameters to tune for GaussianNB
    "LDA": {},
    "QDA": {},
    "MLP (Neural Network)": {
        'hidden_layer_sizes': [(50,), (100,), (100, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001]
    },
    "LightGBM": {
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 200],
        'num_leaves': [31, 50]
    }
}

best_models = {}

for name, model in models.items():
    print(f"\nRunning HalvingGridSearchCV for {name}...")
    params = param_grids.get(name, {})

    if params:
        halving_grid = HalvingGridSearchCV(
            model,
            params,
            scoring='accuracy',
            cv=5,
            factor=2,
            min_resources="exhaust",  # Use as much data as possible
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        halving_grid.fit(X_train, y_train)
        best_models[name] = halving_grid.best_estimator_
        print(f"Best parameters for {name}: {halving_grid.best_params_}")
        print(f"Best cross-validation accuracy: {halving_grid.best_score_:.4f}")
    else:
        model.fit(X_train, y_train)
        best_models[name] = model
        print(f"No parameters to tune for {name}, using default/trained model.")

# Insert the parametes get from from HalvingGridSearchCV

models = {
    "Logistic Regression": LogisticRegression(max_iter = 500, C = 0.1, solver = 'lbfgs'),
    "SVM": SVC(probability=True, C = 10, kernel = 'rbf'),
    "Decision Tree": DecisionTreeClassifier(criterion = 'entropy', max_depth = 7),
    "Random Forest": RandomForestClassifier(n_estimators = 300, max_depth = 20),
    "Gradient Boosting": GradientBoostingClassifier(learning_rate = 0.1, n_estimators = 100),
    "AdaBoost": AdaBoostClassifier(learning_rate = 1, n_estimators = 50),
    "Extra Trees": ExtraTreesClassifier(max_depth = 20, n_estimators = 300),
    "Bagging": BaggingClassifier(max_samples = 1.0, n_estimators = 10),
    "XGBoost": XGBClassifier(learning_rate = 0.1, max_depth = 9, n_estimators = 100, eval_metric='logloss'),
    "KNN": KNeighborsClassifier(n_neighbors = 7),
    "Naive Bayes": GaussianNB(),
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(),
    "MLP (Neural Network)": MLPClassifier(activation = 'relu', alpha = 0.001, hidden_layer_sizes = (100,), max_iter = 500),
    "LightGBM": LGBMClassifier(learning_rate = 0.1, n_estimators = 100, num_leaves = 31),
}

# Train the models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred1 = model.predict(X_test)
    print(f"Classification report for {name}:")
    print(classification_report(y_test, y_pred1))

# Train models and plot ROC curves
plt.figure(figsize=(12, 8))
for name, model in models.items():
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]  # Get probability estimates
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

# Plot settings
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison of Heart Disease Prediction Models')
plt.legend(loc='lower right')
plt.show()

#---------------------------------
# For Validation Dataset
#---------------------------------
val_data = val_data.drop_duplicates()

val_data= val_data.dropna(subset=['target'])

val_data = val_data[val_data['thal'] != 0]

missing_values = val_data.isnull().sum()
total_missing = missing_values.sum()
print("Missing value：", total_missing)

print("Missing values:\n", val_data.isna().sum())

# Replace outliers with NaN
val_data = replace_outliers_with_nan(val_data,numeric_features)

# Identify columns with missing values (including outliers replaced with NaN)
missing_cols_vals = val_data.columns[val_data.isna().sum() > 0]

'''
# Impute Missing Values
for col in missing_cols_vals:
    if val_data[col].isna().sum() > 0:
        round_to_int = col in int_columns
        val_data = rf_impute_missing(val_data, col, train_data, False, round_to_int)
'''

# Impute Outliers
for col in missing_cols_vals:
    if val_data[col].isna().sum() > 0:
        round_to_int = col in int_columns
        val_data = rf_impute_outliers(val_data, col, train_data, False, round_to_int)

val_data = val_data.drop_duplicates()

print("Missing values:\n", val_data.isna().sum())

print("Skewness\n", val_data.skew().sort_values(ascending=False))
#Attributes with highly skewed (fbs, ca) are categorical data

val_data[column_to_transform] = pt.fit_transform(val_data[[column_to_transform]])

# Print transformed and recovered data for verification
print("Transformed 'oldpeak':\n", val_data[[column_to_transform]].head())

print("Skewness of oldpeak after power transform:\n", val_data.skew().sort_values(ascending=False))

val_data[columns_to_scale] = scaler.transform(val_data[columns_to_scale])

val_data[New_age] = MinMax.transform(val_data[New_age])

encoded_val_data = one_hot_encode_columns(val_data, ['cp', 'restecg', 'thal'], category_mapping)
print(encoded_val_data)

encoded_val_data.to_csv('heart_valid.csv',index=False)

'''plot_residuals(encoded_val_data, "target")'''

# Compute the correlation matrix
corr_matrix = encoded_val_data.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))  # Set figure size
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

# Add title
plt.title("Correlation Heatmap")

# Show the plot
plt.show()

result = find_high_correlation(encoded_val_data,  0.7)
for pair in result:
  print(f"Correlation between {pair[0]} and {pair[1]}: {pair[2]:.2f}")

file_path = "/content/heart_valid.csv"
df = pd.read_csv(file_path)

X = df.drop(columns=["target"])
y = df["target"]

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred1 = model.predict(X)
    print(f"Classification report for {name}:")
    print(classification_report(y, y_pred1))

# Train models and plot ROC curves
plt.figure(figsize=(12, 8))
for name, model in models.items():
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X)[:, 1]  # Get probability estimates
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

# Plot settings
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison of Heart Disease Prediction Models')
plt.legend(loc='lower right')
plt.show()

#---------------------------------
# For Testing Dataset
#---------------------------------
test_data = test_data.drop_duplicates()

test_data= test_data.dropna(subset=['target'])

test_data = test_data[test_data['thal'] != 0]

missing_values = test_data.isnull().sum()
total_missing = missing_values.sum()
print("Missing value：", total_missing)

print("Missing values:\n", test_data.isna().sum())

# Replace outliers with NaN
test_data = replace_outliers_with_nan(test_data,numeric_features)

# Identify columns with missing values (including outliers replaced with NaN)
missing_cols_vals = test_data.columns[test_data.isna().sum() > 0]

'''
# Impute Missing Values
for col in missing_cols_vals:
    if test_data[col].isna().sum() > 0:
        round_to_int = col in int_columns
        test_data = rf_impute_missing(test_data, col, train_data, False, round_to_int)
'''

# Impute Outliers
for col in missing_cols_vals:
    if test_data[col].isna().sum() > 0:
        round_to_int = col in int_columns
        test_data = rf_impute_outliers(test_data, col, train_data, False, round_to_int)

test_data = test_data.drop_duplicates()

print("Missing values:\n", test_data.isna().sum())

print("Skewness\n", test_data.skew().sort_values(ascending=False))
#Attributes with highly skewed (fbs, ca) are categorical data

test_data[column_to_transform] = pt.fit_transform(test_data[[column_to_transform]])

# Print transformed and recovered data for verification
print("Transformed 'oldpeak':\n", test_data[[column_to_transform]].head())

print("Skewness of oldpeak after power transform:\n", test_data.skew().sort_values(ascending=False))

test_data[columns_to_scale] = scaler.transform(test_data[columns_to_scale])

test_data[New_age] = MinMax.transform(test_data[New_age])

encoded_test_data = one_hot_encode_columns(test_data, ['cp', 'restecg', 'thal'], category_mapping)
print(encoded_test_data)

encoded_test_data.to_csv('heart_test.csv',index=False)

'''plot_residuals(encoded_test_data, "target")'''

# Compute the correlation matrix
corr_matrix = encoded_test_data.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))  # Set figure size
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

# Add title
plt.title("Correlation Heatmap")

# Show the plot
plt.show()

result = find_high_correlation(encoded_test_data,  0.7)
for pair in result:
  print(f"Correlation between {pair[0]} and {pair[1]}: {pair[2]:.2f}")

file_path = "/content/heart_test.csv"
df = pd.read_csv(file_path)

X = df.drop(columns=["target"])
y = df["target"]

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred1 = model.predict(X)
    print(f"Classification report for {name}:")
    print(classification_report(y, y_pred1))

# Train models and plot ROC curves
plt.figure(figsize=(12, 8))
for name, model in models.items():
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X)[:, 1]  # Get probability estimates
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

# Plot settings
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison of Heart Disease Prediction Models')
plt.legend(loc='lower right')
plt.show()

#---------------------------------
# For Prediction 
#---------------------------------
path = '/content/HeartNewPatients2.csv'
newPatientDf = pd.read_csv(path)

newPatientDf[column_to_transform] = pt.fit_transform(newPatientDf[[column_to_transform]])

newPatientDf[columns_to_scale] = scaler.transform(newPatientDf[columns_to_scale])

newPatientDf[New_age] = MinMax.transform(newPatientDf[New_age])

category_mapping = {
    'cp': ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"],
    'restecg': ["ECG Normal", "ECG Abnormal", "Left Ventricular Hypertrophy"],
    'thal': ["Thalassemia Normal", "Thalassemia Fixed defect", "Reversible defect"]
}

int_to_label_mapping = {
    'cp': {
        0: "Typical angina",
        1: "Atypical angina",
        2: "Non-anginal pain",
        3: "Asymptomatic"
    },
    'restecg': {
        0: "ECG Normal",
        1: "ECG Abnormal",
        2: "Left Ventricular Hypertrophy"
    },
    'thal': {
        1: "Thalassemia Normal",
        2: "Thalassemia Fixed defect",
        3: "Reversible defect"
    }
}


def one_hot_encode_columns_newPatient(df, columns, category_mapping, int_to_label_mapping):
    df_encoded = df.copy()

    for col in columns:
        if col not in df_encoded.columns:
            print(f"Warning: Column '{col}' not in DataFrame. Skipping.")
            continue

        # 1. Convert integers to descriptive labels
        if col in int_to_label_mapping:
            df_encoded[col] = df_encoded[col].map(int_to_label_mapping[col])
        else:
            raise ValueError(f"No int_to_label_mapping provided for column '{col}'")

        # 2. Create encoder with descriptive category names
        encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown='ignore',
            categories=[category_mapping[col]]
        )

        encoded_array = encoder.fit_transform(df_encoded[[col]])
        feature_names = encoder.get_feature_names_out([col])  # Includes names
        feature_names = [name.split('_', 1)[-1] for name in feature_names]

        encoded_array = np.delete(encoded_array, 1, axis=1)
        feature_names.pop(1)  # Remove the second feature name as well

        # 3. Create one-hot encoded DataFrame
        encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=df_encoded.index)

        # 4. Drop original column and join new ones
        df_encoded = df_encoded.drop(columns=[col]).join(encoded_df)

    return df_encoded

encoded_heartNewPatientsDf = one_hot_encode_columns_newPatient(newPatientDf, ['cp', 'restecg', 'thal'], category_mapping, int_to_label_mapping)
print(encoded_heartNewPatientsDf)

encoded_heartNewPatientsDf.to_csv('encodedHeartNewPatient.csv',index=False)

# Load the dataset
file_path = "//content/encodedHeartNewPatient.csv"
encoded_heartNewPatientsDf = pd.read_csv(file_path)

X = encoded_heartNewPatientsDf.drop(columns=["target"])
y = encoded_heartNewPatientsDf["target"]

# Define only the LightGBM model
models = {
    "LightGBM": LGBMClassifier(learning_rate = 0.1, n_estimators = 100, num_leaves = 31),
}

# Deploy the model for real data and print the result
plt.figure(figsize=(12, 8))

# Run only LightGBM model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X)
    print(f"Prediction for model {name}:")
    print(f"{y_pred}")

# Deploy the model for real data and print the result
plt.figure(figsize=(12, 8))

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X)
    print(f"Prediction for model {name}:")
    print(f"{y_pred}")

"""
Feature Preprocessing for Model Training and Inference

Handles data transformation for both logistic regression and XGBoost:
- One-hot encoding + scaling for logistic regression
- Ordinal encoding for XGBoost
- Drops identifiers like policy number and date
- Returns (X, y) during training, or just X for inference
"""

# function 1: prepare for logistic regression

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def prepare_for_logreg(df, return_target=True, return_preprocessor=False):
    ''' 
    During Training, simple call prepare_for_logreg(df)
    During Inference, call prepare_for_logreg(df_new, return_target=False)
    '''
    df = df.copy()
 
    # Separate target if present
    y = None
    if 'fraud_reported' in df.columns:
        y = df['fraud_reported']
        df = df.drop(columns=['fraud_reported'])

    # Drop identifiers / dates
    for col in ['policy_number', 'incident_date']:
        if col in df.columns:
            df = df.drop(columns=col)
    
    # Select column types
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    # Ensure categorical columns are all string
    df[categorical_cols] = df[categorical_cols].astype(str)

    # Define preprocessing pipeline
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

    # Transform features
    X = preprocessor.fit_transform(df)


    if return_preprocessor:
        if return_target:
            return X, y, preprocessor
        else:
            return X, preprocessor
    else:
        if return_target:
            return X, y
        else:
            return X




# function 2: prepare for XGBoost

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

def prepare_for_xgb(df, return_target=True, return_preprocessor=False):
    ''' 
    During Training, simple call prepare_for_logreg(df)
    During Inference, call prepare_for_logreg(df_new, return_target=False)
    '''
    df = df.copy()

    # Separate target if present
    y = None
    if 'fraud_reported' in df.columns:
        y = df['fraud_reported']
        df = df.drop(columns=['fraud_reported'])

    # Drop identifiers / dates
    for col in ['policy_number', 'incident_date']:
        if col in df.columns:
            df = df.drop(columns=col)
    
    # Detect column types
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    # Create preprocessor
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols)
    ])

    X = preprocessor.fit_transform(df)
    feature_names = preprocessor.get_feature_names_out()

    # Return combinations
    if return_preprocessor:
        if return_target:
            return X, y, preprocessor, feature_names
        return X, preprocessor, feature_names
    else:
        if return_target:
            return X, y
        return X






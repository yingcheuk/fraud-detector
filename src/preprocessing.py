''' 
step 1: data cleaning
involve steps like handling missing values, cleaning invalid values, column-retyping, droping useless columns, cross-checking columns, calculate 
'''

import pandas as pd

# function 1: data cleaning

def clean_data(df, verbose=False):

    """
    Clean raw fraud data based on prior exploration.
    - Fixes formatting issues
    - Handles missing values
    - Filters invalid rows
    - Drops unnecessary columns
    """
    
    df = df.copy()

    df_shape_start = df.shape
    
    if verbose:
        print("Starting data cleaning...")

    # Handle inconsistent casing or whitespace
    for col in df.select_dtypes(include='object'):
        df[col] = df[col].astype(str).str.strip().str.lower()
    
    # Target variable
    if 'fraud_reported' in df.columns:
        df['fraud_reported'] = df['fraud_reported'].map({'y': 1, 'n': 0})
        if verbose:
            print("Mapping target column 'fraud_reported' to binary...")

    # Handle missing values - uniform as "Unknown"
    df['authorities_contacted'] = df['authorities_contacted'].fillna('Unknown')
    df['collision_type'] = df['collision_type'].replace('?', 'Unknown')
    df['property_damage'] = df['property_damage'].replace('?', 'Unknown')
    df['police_report_available'] = df['police_report_available'].replace('?', 'Unknown')
    if verbose:
        print("Filling missing values...")

    # Drop rows with invalid date logic
    before_rows = df.shape[0]
    df = df[df['policy_bind_date'] < df['incident_date']]
    after_rows = df.shape[0]
    if verbose:
        print(f"Dropped {before_rows - after_rows} rows with inconsistent dates...")

    # Drop rows where missing dates
    df = df.dropna(subset=['policy_bind_date', 'incident_date'])

    # Drop high-cardinality or useless columns
    df = df.drop(columns=['insured_zip'], errors='ignore')

    # Calculate months_with_policy and replace invalid data in "months_as_customer"
    df['months_with_policy'] = (df['incident_date'] - df['policy_bind_date']) // pd.Timedelta(days=30)
    df.loc[df['months_with_policy'] > df['months_as_customer'], 'months_as_customer'] = df['months_with_policy']
    
    # Drop datetime column after calculation (but keep the 'incident_date' for identification)
    df['incident_month'] = pd.to_datetime(df['incident_date']).dt.month
    df['bind_year'] = pd.to_datetime(df['policy_bind_date']).dt.year
    df = df.drop(columns=['policy_bind_date'], errors='ignore')

    # Retype the columns
    df['policy_deductable'] = df['policy_deductable'].astype(float)
    df['umbrella_limit'] = df['umbrella_limit'].astype(float)
    df['capital-gains'] = df['capital-gains'].astype(float)
    df['capital-loss'] = df['capital-loss'].astype(float)
    df['total_claim_amount'] = df['total_claim_amount'].astype(float)
    df['injury_claim'] = df['injury_claim'].astype(float)
    df['property_claim'] = df['property_claim'].astype(float)
    df['vehicle_claim'] = df['vehicle_claim'].astype(float)

    if verbose:
        print("Finished cleaning...")
        print(f"Shape before cleaning: {df_shape_start}")
        print(f"Shape after cleaning: {df.shape}")

    return df



''' 
step 2: separating inliers and outliers data
'''

# function 2: separating inliers and outliers data

def remove_outliers_iqr(df):
    
    """
    Removes outliers from the given DataFrame using the IQR method on specified numeric columns.
    Returns the cleaned DataFrame and the dropped outliers.
    """
    
    df_clean = df.copy()

    iqr_columns = ['policy_annual_premium',
                     'umbrella_limit',
                     'capital-gains',
                     'capital-loss',
                     'total_claim_amount',
                     'injury_claim',
                     'property_claim',
                     'vehicle_claim']

    for col in iqr_columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    
    # Get dropped outliers
    df_outliers = df.loc[~df.index.isin(df_clean.index)]
    df_inliers = df_clean      
    
    return df_inliers, df_outliers


# function 3: check class balance

def check_class_ratio(df_inliers, df_outliers):
    # Get counts
    inlier_counts = df_inliers["fraud_reported"].value_counts()
    outlier_counts = df_outliers["fraud_reported"].value_counts()

    # Create a summary DataFrame
    summary = pd.DataFrame({
        "Inliers Count": inlier_counts,
        "Outliers Count": outlier_counts
    })

    # Add class ratio (percentage)
    summary["Inliers %"] = (summary["Inliers Count"] / summary["Inliers Count"].sum() * 100).round(2)
    summary["Outliers %"] = (summary["Outliers Count"] / summary["Outliers Count"].sum() * 100).round(2)
    
    summary.index = summary.index.map({0: "Not Fraud", 1: "Fraud"})
    
    print(summary)



















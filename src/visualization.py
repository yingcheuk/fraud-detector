"""
Visualization Tools for XGBoost Models

- Plot feature importance using XGBoost's plot_importance
- Optionally map to original feature names from ColumnTransformer
"""


#function 1: extract real feature names


def get_feature_names_from_column_transformer(ct):
    """
    Extract feature names from a fitted ColumnTransformer.
    """

    if not hasattr(ct, "transformers_"):
        raise TypeError("Expected a ColumnTransformer, but got:", type(ct))
    
    output_features = []

    for name, transformer, columns in ct.transformers_:
        if name == 'remainder' and transformer == 'passthrough':
            output_features.extend(columns)
        elif hasattr(transformer, 'get_feature_names_out'):
            try:
                # Works if transformer stores column context internally
                names = transformer.get_feature_names_out()
            except TypeError:
                # Fallback if transformer needs explicit input
                names = transformer.get_feature_names_out(columns)
            output_features.extend(names)
        else:
            output_features.extend(columns)

    return output_features
    

# function 2: find the top 10 important features

def plot_xgb_importance(model, feature_names=None, max_features=10, importance_type='gain', title=None, save_path=None):
    """
    Plots feature importance with optional custom feature names.

    Parameters:
    - model: trained XGBClassifier model
    - feature_names: list of feature names (optional)
    - max_features: top N features to display
    - importance_type: 'gain', 'weight', 'cover', etc.
    - title: title of the plot (optional)
    - save_path: if given, saves the plot to this path instead of showing
    """
    from xgboost import plot_importance
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, max_features * 0.5))

    # Plot and get axis
    ax = plot_importance(
        model,
        max_num_features=max_features,
        importance_type=importance_type,
        show_values=False,
        xlabel=""
    )

    # Try to map feature names if provided
    if feature_names:
        booster_features = model.get_booster().feature_names
        try:
            mapped_names = []
            for f in booster_features:
                if f.startswith('f') and f[1:].isdigit():
                    idx = int(f[1:])
                    if idx < len(feature_names):
                        mapped_names.append(feature_names[idx])
                    else:
                        mapped_names.append(f)  # fallback
                else:
                    mapped_names.append(f)  # fallback
            ax.set_yticklabels(mapped_names)
        except (IndexError, ValueError, TypeError):
            print("⚠️ Could not map feature names. Using default XGBoost names.")

    if title:
        plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"📸 Plot saved to {save_path}")
    else:
        plt.show()




# function 3: SHAP


def explain_model_with_shap(model, X, feature_names=None, model_name="model"):
    import shap
    import matplotlib.pyplot as plt
    import os

    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    shap.summary_plot(shap_values, X, feature_names=feature_names, plot_size=(7, 5), show=False)

    ax = plt.gca()
    ax.set_title(f"SHAP Summary Plot – {model_name}", fontsize=12)
    ax.tick_params(axis='y', labelsize=9)
    ax.tick_params(axis='x', labelsize=10)

    os.makedirs("/Users/jennifercyc/Desktop/HSH/ML Project/fraud-detector/models/plots", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"/Users/jennifercyc/Desktop/HSH/ML Project/fraud-detector/models/plots/shap_summary_{model_name}.png", dpi=300)
    plt.show()



# function 4: retrieve encoded label values 


def print_category_encoding(preprocessor, col_name):
    """
    Print the label encoding order for a specific categorical column
    from a fitted ColumnTransformer containing an OrdinalEncoder.

    Parameters:
    - preprocessor: fitted ColumnTransformer
    - col_name: string, name of the column to decode
    """
    encoder = preprocessor.named_transformers_['cat']
    col_list = preprocessor.transformers_[1][2]
    idx = col_list.index(col_name)
    print(f"Encoding for '{col_name}': {encoder.categories_[idx]}")



# function 5: find the feature values (categorial)


def get_category_range(preprocessor, col_name, range_type="low"):
    """
    Returns the labels in a specific encoding range for a categorical column.
    range_type: one of ['low', 'midlow', 'mid', 'midhigh', 'high']
    """
    encoder = preprocessor.named_transformers_['cat']
    cats = encoder.categories_[preprocessor.transformers_[1][2].index(col_name)]
    n = len(cats)
    
    quintile = max(1, n // 5)
    ranges = {
        "low": cats[:quintile],
        "midlow": cats[quintile:2*quintile],
        "mid": cats[2*quintile:3*quintile],
        "midhigh": cats[3*quintile:4*quintile],
        "high": cats[4*quintile:]
    }

    print(ranges.get(range_type, []))




# function 6: find the feature values (numerical)


import numpy as np

def get_numerical_range(preprocessor, column_name, range_type="low"):
    """
    Get the approximate original value range for a numerical column 
    based on its scaled distribution using StandardScaler.

    range_type: "low", "midlow", "middle", "midhigh", "high"
    """
    # Access numeric transformer and its columns
    numeric_scaler = preprocessor.named_transformers_['num']
    numeric_cols = preprocessor.transformers_[0][2]

    # Get index and scaler stats
    idx = numeric_cols.index(column_name)
    mean = numeric_scaler.mean_[idx]
    scale = numeric_scaler.scale_[idx]

    # Map range_type to standard deviation range
    ranges = {
        "low": (-np.inf, -1.5),
        "midlow": (-1.5, -0.5),
        "middle": (-0.5, 0.5),
        "midhigh": (0.5, 1.5),
        "high": (1.5, np.inf),
    }
    lower_z, upper_z = ranges[range_type]

    # Inverse transform
    lower_value = mean + lower_z * scale
    upper_value = mean + upper_z * scale

    print((lower_value, upper_value))











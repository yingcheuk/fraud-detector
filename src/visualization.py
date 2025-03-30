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
    
def explain_model_with_shap(model, X, feature_names=None, title=None):
    import shap
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X, feature_names=feature_names, plot_size=(7, 5), show=True)
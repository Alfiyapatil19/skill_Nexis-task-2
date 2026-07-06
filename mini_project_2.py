# Import Libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def main():
    print("=" * 60)
    print(" MINI PROJECT 2: HOUSE PRICE PREDICTION MODEL ")
    print("=" * 60)
    
    # 1. Load Dataset
    dataset_path = "Housing.csv"
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found in the current directory.")
        return
        
    df = pd.read_csv(dataset_path)
    print(f"\n[+] Dataset loaded successfully from '{dataset_path}'")
    print(f"    Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Display first few rows
    print("\n--- First 5 Rows of the Dataset ---")
    print(df.head())
    
    # 2. Check for Missing Values
    print("\n--- Missing Values Check ---")
    missing_vals = df.isnull().sum()
    if missing_vals.sum() == 0:
        print("No missing values found in the dataset.")
    else:
        print(missing_vals[missing_vals > 0])
        
    # 3. Preprocess Categorical Columns
    # Identify non-numeric columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"\nCategorical columns to encode: {categorical_cols}")
    
    # Convert categorical columns to dummy/indicator variables
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    print(f"Dimensions after encoding categorical features: {df_encoded.shape[0]} rows, {df_encoded.shape[1]} columns")
    
    # 4. Define Features (X) and Target (y)
    X = df_encoded.drop("price", axis=1)
    y = df_encoded["price"]
    
    # 5. Train-Test Split
    # Split with 80% train and 20% test, with random_state=42 for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42
    )
    print(f"\nSplit details:")
    print(f" - Training set: {X_train.shape[0]} samples")
    print(f" - Testing set:  {X_test.shape[0]} samples")
    
    # 6. Train Linear Regression Model
    print("\n[+] Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("[+] Model training complete.")
    
    # 7. Predict on Test Set
    print("\n[+] Predicting on the test set...")
    y_pred = model.predict(X_test)
    
    # 8. Evaluate Model Performance
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    
    print("\n" + "=" * 40)
    print("        EVALUATION METRICS")
    print("=" * 40)
    print(f" R² Score (Coefficient of Determination) : {r2:.4f}")
    print(f" Mean Absolute Error (MAE)              : ${mae:,.2f}")
    print(f" Mean Squared Error (MSE)               : {mse:,.2f}")
    print(f" Root Mean Squared Error (RMSE)         : ${rmse:,.2f}")
    print("=" * 40)
    
    # 9. Plot Predicted vs Actual Values
    print("\n[+] Generating Predicted vs Actual scatter plot...")
    plt.figure(figsize=(10, 6))
    
    # Use custom grid style using matplotlib
    plt.grid(True, linestyle='--', alpha=0.5, color='#ccc')
    plt.gca().set_facecolor('#fdfdfd')
    
    # Plot scatter points
    plt.scatter(y_test, y_pred, alpha=0.6, color='#2b5c8f', edgecolors='w', s=60, label="Predictions")
    
    # Plot perfect prediction line (y = x)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='#e05a47', linestyle='--', linewidth=2, label="Perfect Fit (y = x)")
    
    # Formatting
    plt.title("Actual vs. Predicted House Prices", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Actual Price ($)", fontsize=12)
    plt.ylabel("Predicted Price ($)", fontsize=12)
    plt.ticklabel_format(style='plain', axis='both')
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    plt.tight_layout()
    
    # Save the plot
    plot_filename = "predicted_vs_actual.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"[+] Plot saved successfully as '{plot_filename}'")
    plt.close()
    
if __name__ == "__main__":
    main()

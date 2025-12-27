"""
Expense Prediction ML Model
Uses multiple regression algorithms to predict future expenses
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pickle
import warnings
warnings.filterwarnings('ignore')

class ExpensePredictor:
    def __init__(self):
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def create_features(self, df):
        """Create time-based and statistical features from expense data"""
        df = df.copy()
        
        # Time-based features
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week
        
        # Cyclical encoding for month (captures seasonality)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Rolling statistics (if enough data)
        if len(df) > 3:
            df['rolling_mean_3'] = df['amount'].rolling(window=3, min_periods=1).mean()
            df['rolling_std_3'] = df['amount'].rolling(window=3, min_periods=1).std().fillna(0)
        else:
            df['rolling_mean_3'] = df['amount'].mean()
            df['rolling_std_3'] = 0
            
        # Lag features
        df['lag_1'] = df['amount'].shift(1).fillna(df['amount'].mean())
        df['lag_2'] = df['amount'].shift(2).fillna(df['amount'].mean())
        
        # Trend feature
        df['time_index'] = np.arange(len(df))
        
        return df
    
    def prepare_data(self, expenses_data):
        """
        Prepare data for training
        expenses_data: list of dicts with 'date' and 'amount' keys
        Example: [{'date': '2024-01-01', 'amount': 2500}, ...]
        """
        df = pd.DataFrame(expenses_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Create features
        df = self.create_features(df)
        
        # Feature columns for training
        feature_cols = ['month', 'quarter', 'month_sin', 'month_cos', 
                       'rolling_mean_3', 'rolling_std_3', 'lag_1', 'lag_2', 'time_index']
        
        self.feature_names = feature_cols
        
        X = df[feature_cols]
        y = df['amount']
        
        return X, y, df
    
    def train(self, expenses_data):
        """Train all models and select the best one"""
        X, y, df = self.prepare_data(expenses_data)
        
        if len(X) < 5:
            print("Warning: Limited data. Need at least 5 data points for reliable predictions.")
            print("Training with available data, but predictions may be less accurate.")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        if len(X) > 10:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
        else:
            # Use all data for training if dataset is small
            X_train, X_test = X_scaled, X_scaled
            y_train, y_test = y, y
        
        # Train and evaluate all models
        results = {}
        print("\n" + "="*60)
        print("MODEL TRAINING RESULTS")
        print("="*60)
        
        for name, model in self.models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }
            
            print(f"\n{name}:")
            print(f"  MAE:  ₹{mae:.2f}")
            print(f"  RMSE: ₹{rmse:.2f}")
            print(f"  R²:   {r2:.3f}")
        
        # Select best model based on lowest MAE
        self.best_model_name = min(results, key=lambda x: results[x]['mae'])
        self.best_model = results[self.best_model_name]['model']
        
        print("\n" + "="*60)
        print(f"BEST MODEL: {self.best_model_name}")
        print(f"MAE: ₹{results[self.best_model_name]['mae']:.2f}")
        print("="*60 + "\n")
        
        return results
    
    def predict_future(self, expenses_data, months_ahead=3):
        """Predict expenses for future months"""
        _, _, df = self.prepare_data(expenses_data)
        
        predictions = []
        last_date = df['date'].max()
        
        for i in range(1, months_ahead + 1):
            # Create future date
            future_date = last_date + timedelta(days=30 * i)
            
            # Create feature row for prediction
            future_features = {
                'month': future_date.month,
                'quarter': (future_date.month - 1) // 3 + 1,
                'month_sin': np.sin(2 * np.pi * future_date.month / 12),
                'month_cos': np.cos(2 * np.pi * future_date.month / 12),
                'rolling_mean_3': df['amount'].tail(3).mean(),
                'rolling_std_3': df['amount'].tail(3).std(),
                'lag_1': df['amount'].iloc[-1],
                'lag_2': df['amount'].iloc[-2] if len(df) > 1 else df['amount'].iloc[-1],
                'time_index': len(df) + i - 1
            }
            
            # Create feature array
            X_future = pd.DataFrame([future_features])[self.feature_names]
            X_future_scaled = self.scaler.transform(X_future)
            
            # Predict
            predicted_amount = self.best_model.predict(X_future_scaled)[0]
            
            predictions.append({
                'date': future_date,
                'predicted_amount': max(0, predicted_amount)  # Ensure non-negative
            })
        
        return predictions
    
    def plot_predictions(self, expenses_data, predictions):
        """Visualize historical data and predictions"""
        df = pd.DataFrame(expenses_data)
        df['date'] = pd.to_datetime(df['date'])
        
        pred_df = pd.DataFrame(predictions)
        
        plt.figure(figsize=(12, 6))
        
        # Historical data
        plt.plot(df['date'], df['amount'], 'o-', label='Historical', 
                linewidth=2, markersize=8, color='#3b82f6')
        
        # Predictions
        plt.plot(pred_df['date'], pred_df['predicted_amount'], 's--', 
                label='Predicted', linewidth=2, markersize=8, color='#8b5cf6')
        
        plt.xlabel('Date', fontsize=12, fontweight='bold')
        plt.ylabel('Amount (₹)', fontsize=12, fontweight='bold')
        plt.title('Expense Prediction Model', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return plt
    
    def save_model(self, filepath='expense_model.pkl'):
        """Save trained model to disk"""
        model_data = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='expense_model.pkl'):
        """Load trained model from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.best_model = model_data['best_model']
        self.best_model_name = model_data['best_model_name']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        print(f"Model loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    # Sample expense data
    expenses_data = [
        {'date': '2024-01-01', 'amount': 2500},
        {'date': '2024-02-01', 'amount': 2800},
        {'date': '2024-03-01', 'amount': 2600},
        {'date': '2024-04-01', 'amount': 3100},
        {'date': '2024-05-01', 'amount': 2900},
        {'date': '2024-06-01', 'amount': 3200},
        {'date': '2024-07-01', 'amount': 3400},
        {'date': '2024-08-01', 'amount': 3100},
        {'date': '2024-09-01', 'amount': 3300},
        {'date': '2024-10-01', 'amount': 3500},
    ]
    
    # Create and train model
    predictor = ExpensePredictor()
    results = predictor.train(expenses_data)
    
    # Make predictions
    print("\nFUTURE PREDICTIONS:")
    print("-" * 60)
    predictions = predictor.predict_future(expenses_data, months_ahead=3)
    
    for pred in predictions:
        print(f"{pred['date'].strftime('%B %Y')}: ₹{pred['predicted_amount']:.2f}")
    
    # Visualize
    plt = predictor.plot_predictions(expenses_data, predictions)
    plt.savefig('expense_predictions.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'expense_predictions.png'")
    
    # Save model
    predictor.save_model('expense_model.pkl')
    
    print("\n" + "="*60)
    print("Training and prediction completed successfully!")
    print("="*60)
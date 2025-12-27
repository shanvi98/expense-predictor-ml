"""
Script to make predictions using a trained model
"""
from expense_predictor import ExpensePredictor
import pandas as pd

# Load the trained model
predictor = ExpensePredictor()
predictor.load_model('models/expense_model.pkl')

# Load your latest expense data
df = pd.read_csv('data/expenses.csv')
expenses_data = df.to_dict('records')

# Make predictions
months = int(input("How many months ahead to predict? (1-12): "))
predictions = predictor.predict_future(expenses_data, months_ahead=months)

print("\n" + "="*60)
print("EXPENSE PREDICTIONS")
print("="*60)

for pred in predictions:
    print(f"{pred['date'].strftime('%B %Y')}: ₹{pred['predicted_amount']:.2f}")

print("="*60)
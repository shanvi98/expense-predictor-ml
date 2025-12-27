"""
Script to train the expense prediction model with your data
"""
from expense_predictor import ExpensePredictor
import pandas as pd

# Load your expense data
# Option 1: From CSV file
df = pd.read_csv('data/expenses.csv')
expenses_data = df.to_dict('records')

# Option 2: Manually enter data
# expenses_data = [
#     {'date': '2024-01-01', 'amount': 2500},
#     {'date': '2024-02-01', 'amount': 2800},
#     # Add more data...
# ]

# Create and train model
predictor = ExpensePredictor()
print("Training models...")
results = predictor.train(expenses_data)

# Make predictions for next 6 months
print("\n\nPredictions for next 6 months:")
predictions = predictor.predict_future(expenses_data, months_ahead=6)

for pred in predictions:
    print(f"{pred['date'].strftime('%B %Y')}: ₹{pred['predicted_amount']:.2f}")

# Visualize results
plt = predictor.plot_predictions(expenses_data, predictions)
plt.savefig('expense_predictions.png', dpi=300)
print("\nVisualization saved as 'expense_predictions.png'")

# Save the trained model
predictor.save_model('models/expense_model.pkl')
print("Model saved successfully!")
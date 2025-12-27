# 💰 ML Expense Predictor

A machine learning-powered expense prediction system that analyzes your spending patterns and forecasts future expenses using multiple regression algorithms.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

## 🎯 Features

- **5 ML Algorithms**: Automatically trains and compares Linear Regression, Ridge, Lasso, Random Forest, and Gradient Boosting
- **Smart Feature Engineering**: Creates time-based, statistical, and trend features from your data
- **Seasonality Detection**: Captures monthly and quarterly spending patterns
- **Automatic Model Selection**: Picks the best performing model based on your data
- **Visual Predictions**: Generates beautiful charts showing historical data and future predictions
- **Model Persistence**: Save and load trained models for repeated use
- **Confidence Metrics**: Provides MAE, RMSE, and R² scores for model evaluation

## 📋 Requirements

- Python 3.8 or higher
- pip (Python package manager)

## 🚀 Quick Start

### 1. Clone or Download

Download this project to your local machine.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Your Data

Edit `data/expenses.csv` with your expense data:

```csv
date,amount
2024-01-01,2500
2024-02-01,2800
2024-03-01,2600
```

**Important**: 
- Use `YYYY-MM-DD` date format
- Include at least 5-10 data points for reliable predictions
- More data = better predictions!

### 4. Train the Model

```bash
python train_model.py
```

This will:
- Train 5 different ML models on your data
- Compare their performance
- Automatically select the best model
- Generate predictions for the next 6 months
- Save a visualization as `expense_predictions.png`
- Save the trained model to `models/expense_model.pkl`

### 5. Make Predictions

```bash
python predict.py
```

Enter how many months ahead you want to predict (1-12), and the model will generate forecasts using your trained model.

## 📂 Project Structure

```
expense-predictor-ml/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── expense_predictor.py         # Main ML model class
├── train_model.py              # Training script
├── predict.py                  # Prediction script
├── data/
│   └── expenses.csv            # Your expense data
└── models/
    └── expense_model.pkl       # Saved trained model
```

## 📊 Understanding the Output

### Training Output

When you run `train_model.py`, you'll see:

```
============================================================
MODEL TRAINING RESULTS
============================================================

Linear Regression:
  MAE:  ₹125.45
  RMSE: ₹156.78
  R²:   0.892

Random Forest:
  MAE:  ₹98.23
  RMSE: ₹134.56
  R²:   0.925

============================================================
BEST MODEL: Random Forest
MAE: ₹98.23
============================================================
```

**What these metrics mean:**
- **MAE (Mean Absolute Error)**: Average prediction error in rupees (lower is better)
- **RMSE (Root Mean Squared Error)**: Emphasizes larger errors (lower is better)
- **R² Score**: How well the model fits data, 0 to 1 (higher is better, 1 = perfect)

### Prediction Output

```
FUTURE PREDICTIONS:
------------------------------------------------------------
November 2024: ₹3,450.00
December 2024: ₹3,520.00
January 2025: ₹3,380.00
```

## 🔧 Customization

### Change Prediction Period

In `train_model.py`, modify:

```python
predictions = predictor.predict_future(expenses_data, months_ahead=6)  # Change 6 to any number
```

### Use Different Data Source

Instead of CSV, you can directly provide data in `train_model.py`:

```python
expenses_data = [
    {'date': '2024-01-01', 'amount': 2500},
    {'date': '2024-02-01', 'amount': 2800},
    {'date': '2024-03-01', 'amount': 2600},
    # Add more entries...
]
```

### Adjust Model Parameters

In `expense_predictor.py`, modify the model initialization:

```python
self.models = {
    'Random Forest': RandomForestRegressor(
        n_estimators=200,      # Increase trees
        max_depth=10,          # Control tree depth
        random_state=42
    ),
    # Other models...
}
```

## 📈 How It Works

### Feature Engineering

The model creates intelligent features from your data:

1. **Time Features**: Month, quarter, week, day of year
2. **Cyclical Encoding**: Captures seasonality (sin/cos transformations)
3. **Rolling Statistics**: Moving averages and standard deviations
4. **Lag Features**: Previous month expenses
5. **Trend Features**: Time-based indices

### Model Training

1. Loads and preprocesses your expense data
2. Creates engineered features
3. Scales features for optimal performance
4. Trains 5 different ML algorithms
5. Evaluates each model using cross-validation
6. Selects the best performing model
7. Saves the model for future use

### Prediction

1. Loads the trained model
2. Generates features for future dates
3. Makes predictions using the best model
4. Ensures predictions are non-negative
5. Returns forecasted expenses

## 💡 Tips for Best Results

1. **More Data = Better Predictions**: Aim for at least 12 months of data
2. **Regular Updates**: Retrain monthly with new data
3. **Consistent Categories**: Track similar expense types together
4. **Handle Outliers**: Remove or adjust unusual expenses before training
5. **Seasonal Patterns**: The model works best with regular spending patterns

## 🐛 Troubleshooting

### "Limited data" warning
- **Solution**: Add more historical data points (minimum 5, recommended 12+)

### Poor predictions (high MAE)
- **Solution**: Add more data, check for data entry errors, or retrain the model

### Import errors
- **Solution**: Run `pip install -r requirements.txt` again

### Model file not found
- **Solution**: Run `train_model.py` first to create the model

## 📝 Example Workflow

```bash
# 1. Add your expense data to data/expenses.csv

# 2. Train the model
python train_model.py
# Output: Model trained, predictions generated, visualization saved

# 3. View the prediction chart
open expense_predictions.png  # On Mac
# or
start expense_predictions.png  # On Windows

# 4. Make new predictions anytime
python predict.py
# Enter: 3 (for 3 months ahead)
# Output: Predictions for next 3 months
```

## 🔮 Future Enhancements

Potential improvements you can add:

- [ ] Category-wise prediction (groceries, utilities, etc.)
- [ ] Anomaly detection for unusual expenses
- [ ] Web interface for easier interaction
- [ ] Export predictions to Excel/PDF
- [ ] Email alerts for budget overruns
- [ ] Integration with banking APIs

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Feel free to fork this project and submit pull requests with improvements!

## 📧 Support

If you encounter issues or have questions:
1. Check the troubleshooting section above
2. Review the example usage in the code files
3. Ensure your data is in the correct format

## 🙏 Acknowledgments

Built with:
- scikit-learn for machine learning algorithms
- pandas for data manipulation
- matplotlib for visualization
- numpy for numerical operations

---

**Made with ❤️ for better financial planning**

*Last updated: December 2024*

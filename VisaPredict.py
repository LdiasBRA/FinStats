import yfinance as yf
import pandas as pd
import statsmodels.api as sm

data = pd.read_csv("C://Users//lucas//OneDrive//Documents//Econometrics//VisaRegression.csv")

y = data['Close']
x = data[['Volume', 'Open', 'High', 'Low']]

x = sm.add_constant(x)

model = sm.OLS(y,x).fit()

print(model.summary())



# Fetch recent data for Visa
visa_data = yf.download('V', start='2014-03-04', end='2024-03-05')

# Assume you want to predict the closing price for the latest available data
# Prepare the data
latest_data = visa_data.iloc[-1][['Volume', 'Open', 'High', 'Low']].to_frame().T

# Add a constant to the new data
latest_data_with_constant = sm.add_constant(latest_data, has_constant='add')

# Load your model (assuming it's saved or you have the coefficients)
# For demonstration, let's say you have the model loaded as `model`

# Predict using the model
predicted_close = model.predict(latest_data_with_constant)

print("Predicted closing price for Visa:", predicted_close)

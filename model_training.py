import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
from car_data_prep import prepare_data

# קריאת הנתונים
data = pd.read_csv('dataset.csv')

# הכנת הנתונים
data_prepared = prepare_data(data)
print("Training columns:", data_prepared.columns)  # הוספת הדפסה של העמודות לאחר העיבוד

# נרמול הנתונים
scaler = StandardScaler()
X = data_prepared.drop('Price', axis=1)
y = data_prepared['Price']
X_scaled = scaler.fit_transform(X)

# שמירת שמות העמודות לאחר העיבוד
columns = data_prepared.columns
joblib.dump(columns, 'model_columns.pkl')
joblib.dump(scaler, 'scaler.pkl')  # שמירת הסקיילר

# חלוקת הנתונים ל-Train ו-Test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# אימון המודל
model = RandomForestRegressor()
model.fit(X_train, y_train)

# שמירת המודל
joblib.dump(model, 'trained_model.pkl')

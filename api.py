from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from car_data_prep import prepare_data

app = Flask(__name__)
model = joblib.load('trained_model.pkl')
model_columns = joblib.load('model_columns.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # קבלת נתונים מהטופס
    data = request.form.to_dict()
    print("Input data:", data)  # הדפסה של הנתונים מהטופס
    df = pd.DataFrame([data])

    # עיבוד הנתונים
    df_prepared = prepare_data(df, is_training=False)
    print("Prediction columns before reindex:", df_prepared.columns)  # הדפסה של העמודות לפני reindex

    # הבטחת שהעמודות מתאימות למודל
    df_prepared = df_prepared.reindex(columns=[col for col in model_columns if col != 'Price'], fill_value=0)
    print("Prediction columns after reindex:", df_prepared.columns)  # הדפסה של העמודות אחרי reindex

    # נרמול הנתונים
    df_prepared_scaled = scaler.transform(df_prepared)
    print("Prepared data for prediction:", df_prepared_scaled)  # הדפסה של הנתונים המעובדים

    # חיזוי מחיר הרכב
    try:
        prediction = model.predict(df_prepared_scaled)
        print("Prediction:", prediction)
        return jsonify({'predicted_price': prediction[0]})
    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

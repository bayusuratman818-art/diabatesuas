from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# =====================
# LOAD MODEL
# =====================

model = joblib.load(
    "model_diabetes.pkl"
)

encoder_gender = joblib.load(
    "encoder_gender.pkl"
)

encoder_smoking = joblib.load(
    "encoder_smoking.pkl"
)

# =====================
# HOME
# =====================

@app.route('/')
def home():

    return render_template(
        'index.html'
    )

# =====================
# PREDIKSI
# =====================

@app.route(
    '/predict',
    methods=['POST']
)

def predict():

    try:

        gender = request.form['gender']

        age = float(
            request.form['age']
        )

        hypertension = int(
            request.form[
                'hypertension'
            ]
        )

        heart = int(
            request.form[
                'heart_disease'
            ]
        )

        smoking = request.form[
            'smoking_history'
        ]

        bmi = float(
            request.form['bmi']
        )

        hba1c = float(
            request.form[
                'hba1c'
            ]
        )

        glucose = float(
            request.form[
                'glucose'
            ]
        )

        gender = (
            encoder_gender
            .transform(
                [gender]
            )[0]
        )

        smoking = (
            encoder_smoking
            .transform(
                [smoking]
            )[0]
        )

        data = np.array([[
            gender,
            age,
            hypertension,
            heart,
            smoking,
            bmi,
            hba1c,
            glucose
        ]])

        prediction = (
            model.predict(
                data
            )[0]
        )

        prob = (
            model.predict_proba(
                data
            )[0]
        )

        normal_prob = round(
            prob[0]*100,
            2
        )

        diabetes_prob = round(
            prob[1]*100,
            2
        )

        result = (
            "Positif Diabetes"
            if prediction == 1
            else
            "Tidak Diabetes"
        )

        risk = (
            "high"
            if prediction == 1
            else
            "low"
        )

        return render_template(

            "index.html",

            prediction=result,

            risk=risk,

            normal_prob=normal_prob,

            diabetes_prob=diabetes_prob,

            form_data=request.form
        )

    except Exception as e:

        return render_template(
            "index.html",

            error=str(e)
        )


if __name__=="__main__":

    app.run(
        debug=True
    )
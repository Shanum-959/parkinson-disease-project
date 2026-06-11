from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import librosa
import os

app = Flask(__name__)

# Models load karo
model_tabular = joblib.load('models/model_tabular.pkl')
scaler_tabular = joblib.load('models/scaler_tabular.pkl')
model_voice = joblib.load('models/model_voice.pkl')
scaler_voice = joblib.load('models/scaler_voice.pkl')

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Tabular prediction
@app.route('/tabular', methods=['GET', 'POST'])
def tabular():
    if request.method == 'POST':
        data = request.form
        features = pd.DataFrame([{
            'Age': float(data['age']),
            'Gender': int(data['gender']),
            'Ethnicity': int(data['ethnicity']),
            'EducationLevel': int(data['education']),
            'BMI': float(data['bmi']),
            'Smoking': int(data['smoking']),
            'AlcoholConsumption': float(data['alcohol']),
            'PhysicalActivity': float(data['physical']),
            'DietQuality': float(data['diet']),
            'SleepQuality': float(data['sleep']),
            'FamilyHistoryParkinsons': int(data['family']),
            'TraumaticBrainInjury': int(data['tbi']),
            'Hypertension': int(data['hypertension']),
            'Diabetes': int(data['diabetes']),
            'Depression': int(data['depression']),
            'Stroke': int(data['stroke']),
            'SystolicBP': float(data['systolic']),
            'DiastolicBP': float(data['diastolic']),
            'CholesterolTotal': float(data['chol_total']),
            'CholesterolLDL': float(data['chol_ldl']),
            'CholesterolHDL': float(data['chol_hdl']),
            'CholesterolTriglycerides': float(data['chol_tri']),
            'UPDRS': float(data['updrs']),
            'MoCA': float(data['moca']),
            'FunctionalAssessment': float(data['functional']),
            'Tremor': int(data['tremor']),
            'Rigidity': int(data['rigidity']),
            'Bradykinesia': int(data['bradykinesia']),
            'PosturalInstability': int(data['postural']),
            'SpeechProblems': int(data['speech']),
            'SleepDisorders': int(data['sleep_disorders']),
            'Constipation': int(data['constipation']),
        }])

        scaled = scaler_tabular.transform(features)
        pred = model_tabular.predict(scaled)[0]
        prob = model_tabular.predict_proba(scaled)[0]
        confidence = round(max(prob) * 100, 1)
        result = "Parkinson's Detected" if pred == 1 else "Healthy"
        return render_template('tabular.html', result=result, confidence=confidence)

    return render_template('tabular.html', result=None)

# Voice prediction
@app.route('/voice', methods=['GET', 'POST'])
def voice():
    if request.method == 'POST':
        file = request.files['audio']
        path = 'temp_audio.wav'
        file.save(path)

        audio, sr = librosa.load(path, sr=22050, mono=True)
        f0, _, _ = librosa.pyin(audio, fmin=70, fmax=500, sr=sr)
        f0 = f0[~np.isnan(f0)]

        f0_mean = np.mean(f0) if len(f0) > 0 else 150
        f0_max  = np.max(f0)  if len(f0) > 0 else 200
        f0_min  = np.min(f0)  if len(f0) > 0 else 100
        jitter  = np.std(np.diff(f0)) / f0_mean if len(f0) > 1 else 0.005
        jitter_abs = np.mean(np.abs(np.diff(f0))) / 1000 if len(f0) > 1 else 0.00005
        rap = jitter / 3
        ppq = jitter / 2
        ddp = jitter * 3
        rms = librosa.feature.rms(y=audio)[0]
        rms_mean = np.mean(rms)
        shimmer    = np.std(np.diff(rms)) / rms_mean if rms_mean > 0 else 0.05
        shimmer_db = 20 * np.log10(shimmer + 1e-8)
        apq3 = shimmer / 3
        apq5 = shimmer / 2
        apq  = shimmer * 1.5
        dda  = shimmer * 3
        zcr  = librosa.feature.zero_crossing_rate(audio)[0].mean()
        nhr  = zcr / (rms_mean + 1e-8)
        hnr  = 20 * np.log10(rms_mean / (zcr + 1e-8))
        stft = np.abs(librosa.stft(audio))
        energy = np.mean(stft, axis=1)
        rpde = np.std(energy) / (np.mean(energy) + 1e-8)
        dfa  = np.corrcoef(np.arange(len(rms)), rms)[0, 1] if len(rms) > 1 else 0.6
        spread1 = -np.var(f0) if len(f0) > 0 else -5
        spread2 =  np.std(f0) if len(f0) > 0 else 0.2
        d2  = rpde * 2
        ppe = np.std(f0) / f0_mean if len(f0) > 0 else 0.2

        features = np.array([[
            f0_mean, f0_max, f0_min,
            jitter, jitter_abs, rap, ppq, ddp,
            shimmer, shimmer_db, apq3, apq5, apq, dda,
            nhr, hnr, rpde, dfa,
            spread1, spread2, d2, ppe
        ]])

        scaled = scaler_voice.transform(features)
        pred = model_voice.predict(scaled)[0]
        prob = model_voice.predict_proba(scaled)[0]
        confidence = round(max(prob) * 100, 1)
        result = "Parkinson's Detected" if pred == 1 else "Healthy"
        os.remove(path)
        return render_template('voice.html', result=result, confidence=confidence)

    return render_template('voice.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)
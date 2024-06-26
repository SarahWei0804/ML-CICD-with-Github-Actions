import gradio as gr
import skops.io as sio

pipe = sio.load("model/heart_pipeline.skops", trusted=True)

def predict_heart(age, sex, chestpaintype, restingbp, cholesterol, fastingbs, restingecg, maxhr, exerciseangina, oldpeak, st_slope):
    """Predict heart disease based on patient features.
    Args:
        age: age of aptient
        sex: sex of patient [M: Male, F: Female]
        chestpaintype: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
        restingbp: resting blood pressure [mm Hg]
        cholesterol: serum cholesterol [mm/dl]
        fastingbs: fasting blood sugar [1: if fastingbs > 120 mg/dl, o: otherwise]
        restingecg: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
        maxhr: maximum heart rate achieved [Numeric value between 60 and 202]
        exerciseangina: exercise-induced angina [Y: Yes, N: No]
        oldpeak: oldpeak = ST [Numeric value measured in depression]
        st_slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
    """
    features = [age, sex, chestpaintype, restingbp, cholesterol, fastingbs, restingecg, maxhr, exerciseangina, oldpeak, st_slope]
    predicted_heart = pipe.predict([features])[0]
    label = f"Predicted Heart disease: {predicted_heart}"
    return label

inputs = [gr.Slider(0, 105, step=1, label="Age"),
          gr.Radio(['M','F'], label='Sex'),
          gr.Radio(['TA', 'ATA', 'NAP', 'ASY'], label='ChestPainType'),
          gr.Slider(0,300, step=1, label='RestingBP'),
          gr.Slider(0, 1000, step=1, label='Cholesterol'),
          gr.Radio(['1', '0'], label='FastingBS'),
          gr.Radio(['Normal', 'ST', 'LVH'], label='RestingECG'),
          gr.Slider(60, 202, step=1, label='MaxHR'),
          gr.Radio(['Y', 'N'], label='ExerciseAngina'),
          gr.Slider(0,500,step=1, label='Oldpeak'),
          gr.Radio(['Up', 'Flat', 'Down'], label='ST_Slope')]
outputs = [gr.Label(num_top_classes=2)]

examples = [[20, 'F', 'Typical Angina', 80, 200, 'over 120', 'Normal', 90, 'Y', 60, 'Flat']]

title = 'Heart Disease classification'
description = 'Enter the details to identify the possibility to have heart disease.'

gr.Interface(fn=predict_heart,
             inputs=inputs,
             outputs=outputs,
             examples=examples,
             title=title,
             description=description,
             theme=gr.themes.Soft()
            ).launch(share = True)
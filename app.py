from flask import Flask, render_template
from flask import request
import os
import pickle
from keras.models import load_model
import cv2
import numpy as np

app = Flask(__name__)

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH, 'static//uploaded//')
MODEL_PATH = os.path.join(BASE_PATH, 'static//MODELS//')

# ------- Loading model----------
xray_model_path = os.path.join(MODEL_PATH, 'xray.model')
symptom_model_path = os.path.join(MODEL_PATH, 'model.save')
xray_model = load_model(xray_model_path)
symptom_model = pickle.load(open(symptom_model_path, 'rb'))


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        #  taking the image from the website
        upload_file = request.files['image_name']
        filename = upload_file.filename
        print('The filename that has been uploaded is ', filename)

        # checking the extension of the image uploaded
        ext = filename.split('.')[-1]
        if ext.lower() in ['jpg', 'jpeg', 'png']:
            # saving the image
            path_save = os.path.join(UPLOAD_PATH, filename)
            upload_file.save(path_save)
            print('File saved successfully!')
            # taking the symptoms data from the website
            symptoms = [convert(request.form.getlist('mycheckbox'))]
            #symptoms = np.array(symptoms)
            # print(symptoms)

            # ------calling the symptoms model
            sym_result = symp(symptoms, symptom_model)

            # ---------- calling the model---------------------------
            xray_result = xray(path_save, xray_model)

            # ------------Deciding code of the output----------------
            if xray_result == 'VIRAL PNEUMONIA' and sym_result == 'NO COVID':
                RE = 'VIRAL PNEUMONIA'
                descript = 'VIRAL PNEUMONIA : Follow the doctor instructions properly, stay home, stay safe and help to control and win over the covid.'
            elif xray_result == 'VIRAL PNEUMONIA' and sym_result == 'COVID':
                RE = 'VIRAL PNEUMONIA'
                descript = 'VIRAL PNEUMONIA : Follow the doctor instructions properly, symptoms matching with covid, strictly avoid public places and help to control and win over the covid.'
            elif xray_result == 'NORMAL' and sym_result == 'NO COVID':
                RE = 'NORMAL'
                descript = 'COVID NEGATIVE : Normal reports, keep yourself hygiene and help to control and win over the covid.'
            elif xray_result == 'NORMAL' and sym_result == 'COVID':
                RE = 'NORMAL'
                descript = 'COVID NEGATIVE : But your symptoms matches with covid please isolate yourself from other and help to control and win over the covid.'
            elif xray_result == 'COVID' and sym_result == 'NO COVID':
                RE = 'COVID'
                descript = 'COVID POSITIVE : Please follow the covid restrictions strictly and help to control and win over the covid.'
            elif xray_result == 'COVID' and sym_result == 'COVID':
                RE = 'COVID'
                descript = 'COVID POSITIVE : Please follow the covid restrictions strictly and help to control and win over the covid.'

            return render_template('upload.html', fileupload=True, data=[RE, descript], image=filename)

        else:
            print('Use only the extension of jpg, jpeg or png')
            return render_template('upload.html')

    else:
        return render_template('upload.html', fileupload=False)

# Xray model takes the image


def xray(path, model):
    testing = cv2.imread(path)
    resized = cv2.resize(testing, (100, 100))

    normalized = resized/255.0
    reshaped = np.reshape(normalized, (1, 100, 100, 3))
    result = model.predict(reshaped)
    dic = {0: 'VIRAL PNEUMONIA', 1: 'NORMAL', 2: 'COVID'}
    R = np.argmax(result)
    return dic[R]

# symptom model takes the symptoms


def symp(s, model):
    print(s)
    re = model.predict(s)
    if re[0] == 1:
        return 'COVID'
    if re[0] == 0:
        return 'NO COVID'


def convert(li):
    fi = []
    for i in range(10):
        if str(i) in li:
            fi.append(1)
        else:
            fi.append(0)
    return fi


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5500, debug=True)
    # app,run(debug=True)

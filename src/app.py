from flask import Flask, render_template, request
import base64
import skimage.io
from skimage.color import rgb2gray
from skimage.transform import resize
import numpy as np
import json
from keras import Sequential, Model, Input
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, concatenate
from keras import backend as K
import cv2



app = Flask(__name__)

def byte_data_to_numpy(byte_data):
    raw_data = base64.b64decode(byte_data)
    raw_data = skimage.io.imread(raw_data, plugin='imageio')
    gray_data = rgb2gray(raw_data)
    return gray_data

def return_face(gray_data, rescale_size=(48, 48)):
    gray_data *= 255
    gray_data = gray_data.astype(np.uint8)
    face_detector = cv2.CascadeClassifier(r'static/assets/haarcascade_frontalface_default.xml')
    try:
        bounding_rectangle = face_detector.detectMultiScale(gray_data, 1.3, 5)[0]
    except:
        return False
    face = gray_data[bounding_rectangle[1]:(bounding_rectangle[1] + bounding_rectangle[3]),
           bounding_rectangle[0]:(bounding_rectangle[0] + bounding_rectangle[2])]
    face = face/255
    face = resize(face, (rescale_size[0], rescale_size[1]))
    return(face)

def load_model():
    input_shape = (48, 48, 1)
    kernel_sizes = [(2, 2), (3, 3), (4, 4), (5, 5)]
    convs = []
    inp = Input(shape=input_shape)

    for k in range(len(kernel_sizes)):
        conv = Conv2D(16, kernel_sizes[k], padding='same',
                      activation='relu')(inp)
        convs.append(conv)

    concatenated = concatenate(convs, axis=1)
    concatenated = MaxPool2D((2, 2), strides=(2, 2))(concatenated)
    concatenated = Conv2D(64, (3, 3), activation='relu', padding='same')(concatenated)
    concatenated = MaxPool2D((2, 2), strides=(2, 2))(concatenated)
    concatenated = Conv2D(64, (3, 3), activation='relu', padding='same')(concatenated)
    concatenated = MaxPool2D((2, 2), strides=(2, 2))(concatenated)

    flat = Flatten()(concatenated)
    d1 = Dense(1000, activation='relu')(flat)
    d1 = Dropout(.3)(d1)
    d2 = Dense(1000, activation='relu')(d1)
    d2 = Dropout(.3)(d2)
    d3 = Dense(500, activation='relu')(d2)
    d3 = Dropout(.3)(d3)
    out = Dense(7, activation='softmax')(d3)

    model = Model(inp, out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    model.load_weights(r'static/assets/stack_mod1.h5')

    return (model)


@app.route('/', methods=["GET", "POST"])
def snap_pic():
    if request.method == "GET":
        return render_template('webcam.html')
    else:
        byte_data = request.form['capture_image']
        gray_data = byte_data_to_numpy(byte_data)
        face = return_face(gray_data)
        if type(face) is not np.ndarray:
            return render_template("error.html")
        z = json.dumps(np.flip(face, 0).tolist())
        model = load_model()
        prediction_probs = model.predict(face.reshape(1, 48, 48, 1))
        prediction_map = ['Angry', 'Disgusted', 'Afraid', 'Happy', 'Sad', 'Surprised', 'Neutral']
        prediction = prediction_map[np.argmax(prediction_probs)]
        prediction_probs = json.dumps(prediction_probs.tolist()[0])
        K.clear_session()
        return render_template("result.html",
                               z=z,
                               prediction_probs=prediction_probs,
                               prediction=prediction)

if __name__ == '__main__':
    app.run()

import os
from uuid import uuid4

from flask import Flask, request, render_template, send_from_directory,flash
import pandas as pd
import csv
from tensorflow.keras.models import load_model
import numpy as np
import string
import random, copy
import io
from PIL import Image
import re
import base64
import PIL
import cv2

import math
import random





# from flask import Flask, request,flash,render_template,send_from_directory,redirect,url_for
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.models import load_model
# import numpy as np
# import os
# import pandas as pd
# import cv2
# import matplotlib
# matplotlib.use('Agg')
# from pylab import*


app=Flask(__name__)
app.secret_key='random string'

classes=['Normal:','Pneumonia:']


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload/<filename>')
def send_image(filename):
    print('kjsifhuissywudhj')
    return send_from_directory("images", filename)

@app.route("/upload", methods=["POST","GET"])
def upload():
    print('a')
    m = int(request.form["alg"])

    myfile = request.files['file']
    fn = myfile.filename
    mypath = os.path.join('images/', fn)
    myfile.save(mypath)

    print("{} is the file name", fn)
    print("Accept incoming file:", fn)
    print("Save it to:", mypath)
    # import tensorflow as tf
    import numpy as np
    from tensorflow.keras.preprocessing import image
    acc = pd.read_csv("Acc.csv")
    from tensorflow.keras.models import load_model
    if m == 1:
        new_model = load_model(r'visualizations\CNN.h5')
        test_image = image.load_img(mypath, target_size=(64,64))
        test_image = image.img_to_array(test_image)
        a = acc.iloc[m - 1, 1]
        
    else:
        new_model = load_model(r'visualizations\RNN.h5')
        test_image = image.load_img(mypath, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = test_image / 255.0
        # a = acc.iloc[m - 1, 1]
    # mypath="data/train/NORMAL/IM-0117-0001.jpeg"
    # new_model.summary()
    # test_image = image.load_img('D:\\image classification\\images\\'+filename,target_size=(64,64))
    # test_image=test_image/255.0
    test_image = np.expand_dims(test_image, axis=0)
    result = new_model.predict(test_image)

    prediction = classes[np.argmax(result)]
    # result=np.argmax(prediction, axis=1)[0]
    # accuracy=float(np.max(prediction, axis=0)[0])

    return render_template("template.html", text=prediction,image_name=fn)


if __name__=='__main__':
    app.run(debug=True)
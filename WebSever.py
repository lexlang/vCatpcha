import os,io,base64
from keras_segmentation.models.unet import resnet50_unet
from flask import Flask,request,jsonify
import numpy as np
import cv2

application=Flask(__name__)
model = resnet50_unet(n_classes=2,  input_height=256, input_width=512)
model.load_weights("58.h5")

@application.route('/localOcr',methods=['POST'])
def recognition():
    imgBase64 = str(request.get_data()).replace("b'image=","")#直接数据的原始结构
    imgBase64 =  base64.b64decode(imgBase64)
    imgBase64 = np.frombuffer(imgBase64, np.uint8)
    imgBase64 =  cv2.imdecode(imgBase64, cv2.IMREAD_ANYCOLOR)
    out = model.predict_segmentation(
        inp=imgBase64
        ,out_fname = "3.png"
    )
    return jsonify(out.tolist())

if __name__=="__main__":
    application.run(host='0.0.0.0', port=80, debug=True, use_reloader=False)
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from flask import send_from_directory
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = tf.keras.models.load_model("rice_disease_fixed.h5")

classes = [
"Healthy",
"Leaf Scald Disease", 
"Sheath Blight Disease",
"Bacterial Leaf Blight Disease",
"Rice Blast Disease",
"Brown Spot Disease",
"Unknown",
]

disease_info = {

"Rice Blast Disease":{
"cause":"เกิดจากเชื้อรา Pyricularia oryzae ทำให้เกิดแผลรูปตา",
"treatment":"ใช้พันธุ์ต้านทาน และพ่นสารป้องกันเชื้อรา"
},

"Brown Spot Disease":{
"cause":"เกิดจากเชื้อรา Bipolaris oryzae ทำให้เกิดจุดสีน้ำตาล",
"treatment":"ปรับปรุงดิน และใช้สารป้องกันเชื้อรา"
},

"Leaf Scald Disease":{
"cause":"เกิดจากเชื้อรา Microdochium oryzae",
"treatment":"ลดความชื้น และใช้สารกำจัดเชื้อรา"
},

"Sheath Blight Disease":{
"cause":"เกิดจากเชื้อรา Rhizoctonia solani",
"treatment":"ลดความหนาแน่นต้นข้าว และใช้สารป้องกันเชื้อรา"
},

"Bacterial Leaf Blight Disease":{
"cause":"เกิดจากแบคทีเรีย Xanthomonas oryzae",
"treatment":"ใช้พันธุ์ต้านทาน และจัดการน้ำในนา"
},

"Healthy":{
"cause":"ใบข้าวปกติ ไม่พบโรค",
"treatment":"ดูแลสภาพแวดล้อมให้เหมาะสม"
},

"Unknown":{
"cause":"ไม่สามารถระบุโรคได้",
"treatment":"กรุณาตรวจสอบภาพใหม่ หรือใช้ภาพที่ชัดเจนกว่า"
}

}

def predict(img_path):

    img = Image.open(img_path).resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img,axis=0)

    prediction = model.predict(img)

    index = np.argmax(prediction)

    confidence = float(np.max(prediction))*100

    disease = classes[index]

    return disease, round(confidence,2)


@app.route("/",methods=["GET","POST"])
def index():

    prediction=None
    confidence=None
    cause=None
    treatment=None
    img_path=None

    if request.method=="POST":

        file=request.files["file"]

        if file:

            path=os.path.join(UPLOAD_FOLDER,file.filename)
            file.save(path)

            disease,conf=predict(path)

            prediction=disease
            confidence=conf

            cause=disease_info[disease]["cause"]
            treatment=disease_info[disease]["treatment"]

            img_path=path

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence,
                           cause=cause,
                           treatment=treatment,
                           img_path=img_path)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
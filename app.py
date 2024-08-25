#imports
import io
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from PIL import Image 

#create the app object
app = FastAPI()

#load models
model1 = tf.keras.models.load_model("./model/class1.h5",compile=False)
model2 = tf.keras.models.load_model("./model/class2.h5",compile=False)
model3 = tf.keras.models.load_model("./model/class3.h5",compile=False)
model4 = tf.keras.models.load_model("./model/class4.h5",compile=False)
model5 = tf.keras.models.load_model("./model/class5.h5",compile=False)
model6 = tf.keras.models.load_model("./model/class6.h5",compile=False)

model_class2 = tf.keras.models.load_model("./model/2Wobot1_train_model.h5")

def preprocess_image(image: Image.Image) -> np.ndarray:
    # Resize, normalize, or do any other preprocessing required by your model
    image = image.resize((512, 512))  # Example size, adjust as needed
    image_array = np.array(image) / 255.0  # Normalize to [0,1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

@app.get('/')
def home():
    return {'Surface Defect Detection Application Using Image Segmentation'}

# All logic for handling the image segmentation.
# It requires the desired model and the image in which to perform object detection.
@app.post('/predict')
async def predict(file: UploadFile = File(...)):

    # Read and open the image file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    img = preprocess_image(image)

    ac1 = model1.predict(img)
    ac2 = model2.predict(img)
    ac3 = model3.predict(img)
    ac4 = model4.predict(img)
    ac5 = model5.predict(img)
    ac6 = model6.predict(img)

    predictions = model_class2.predict(img)

    #predictions
    score = tf.nn.softmax(predictions)
    scorearr = np.array(score)
    x = scorearr.argmax(axis = 1)[:,None]

    if(x==0):
        predict = ac1[0,:,:,0]
    elif(x==1):
        predict = ac2[0,:,:,0]
    elif(x==2):
        predict = ac3[0,:,:,0]
    elif(x==3):
        predict = ac4[0,:,:,0]
    elif(x==4):
        predict = ac5[0,:,:,0]
    elif(x==5):
        predict = ac6[0,:,:,0]
    else:
        print("Error")

    predarr = np.array(predict)

    col = Image.fromarray(predarr*255)
    gray = col.convert('L')

    # Let numpy do the heavy lifting for converting pixels to pure black or white
    bw = np.asarray(gray).copy()

    # Pixel range is 0...255, 256/2 = 128
    bw[bw < 128] = 0   # Black
    bw[bw >= 128] = 255 # White

    output_image = Image.fromarray(bw)
    
    # 4. STREAM THE RESPONSE BACK TO THE CLIENT
    byte_io = io.BytesIO()
    output_image.save(byte_io, format='PNG')
    byte_io.seek(0)

    # Return the image as a streaming response
    return StreamingResponse(byte_io, media_type="image/png")

    # Open the saved image for reading in binary mode
    file_image = open(f'images_uploaded/{filename}', mode="rb")
    
    # Return the image as a stream specifying media type
    return StreamingResponse(file_image, media_type="image/jpeg")

if __name__ == '__main__':
    uvicorn.run(app, host= '127.0.0.1', port= 8000)
#uvicorn main:app --reload
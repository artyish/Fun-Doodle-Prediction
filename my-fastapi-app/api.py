from fastapi import FastAPI , File, UploadFile
from predict import predict_doodle
from PIL import Image
from io import BytesIO
import numpy as np
from fastapi.middleware.cors import CORSMiddleware


#origins = [
#    "http://localhost:3000",  # your frontend URL
#    "http://127.0.0.1:3000",
#    # add other origins you want to allow
#]

api = FastAPI()

# GET - to get information
# POST - submitting new information
# PUT - updating information 
# DELETE - deleting information

api.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:3000","http://localhost:3000","https://hoppscotch.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@api.get('/')
def index():
    return {"message":"Hello World"}


@api.post('/predict')
async def predict(file: UploadFile = File(...)):
    content = await file.read() # wait for the file to be read first
    image = Image.open(BytesIO(content)).convert('L') # bytesIO to handle bytes 
    img_array = np.array(image) # because it is converted through PIL we have to convert it to nparray
    get_response = predict_doodle(img_array) # send it to the other file
    return get_response



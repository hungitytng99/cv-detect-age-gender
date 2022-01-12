# import necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
import cvlib as cv
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle

# download pre-trained model_gender file (one-time download)
# model_gender_path = "./models/gender_detection.model" 
# model_gender = load_model(model_gender_path)

# download pre-trained model_gender file (one-time download)
model_age_path = "models/rfc_canny_model_acc_0.618" 
model_age = pickle.load(open(model_age_path, 'rb'))

# load the model from disk
# open webcam
webcam = cv2.VideoCapture(0)
    
# gender_classes = ['Male','Female']
age_classes=['1-18', '19-29', '30-49', '50-116']

def give_col_image(path):
    img = cv2.imread(path)
    col_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return col_img

def give_gray_image(col_img):
    gray_img = cv2.cvtColor(col_img, cv2.COLOR_BGR2GRAY)
    return gray_img

def give_canny_image(gray_img):
    canny_img = cv2.Canny(gray_img, sigma=0.9)
    return canny_img


def features_grid(img):
    features = np.array([], dtype='uint8')
    section = 1
    for y in range(0, img.shape[0], 10):
        for x in range(0, img.shape[1], 10):
            # Cropping the image into a section.
            section_img = img[y:y+10, x:x+10]
            # Claculating the mean and stdev of the sectioned image.
            section_mean = np.mean(section_img)
            section_std = np.std(section_img)
            # Appending the above calculated values into features array.
            features = np.append(features, [section_mean, section_std])
    # Returning the features array.
    return features


def rf_predict_age(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Converting the coloured image to a grayscale image.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Converting the grayscale image to a canny edges filtered image.
    img = cv2.Canny(img,80,180)
    # Using the function defined above, extracting the features (mean and stdev values of all 10x10 pixel sections from the image) from the canny edges filtered image.
    img_features = features_grid(img)
    img_features = img_features.reshape(1, img_features.shape[0])
    idx = model_age.predict(img_features)[0]
    
   
    print("PREDICT AGE:" , idx)
    label = age_classes[idx]
    label = "{}".format(label)
    return label

def showWebcam(webcam):
    status, frame = webcam.read()
    face, confidence = cv.detect_face(frame)
    # loop through detected faces
    for idx, f in enumerate(face):
        # get corner points of face rectangle        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]
        # draw rectangle over face
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)
        # crop the detected face region
        face_crop = np.copy(frame[startY:endY,startX:endX])
        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        tra_face_crop = cv2.resize(face_crop, (200,200))
        # Converting the coloured image to a grayscale image.
        face_crop = cv2.resize(face_crop, (96,96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        Y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(frame, "RF (4 classes):" + rf_predict_age(tra_face_crop), (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
    return frame
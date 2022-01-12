# import necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
import cvlib as cv

# download pre-trained model_gender file (one-time download)
model_age_path = "models/age_CNN_0.69_4.h5" 
model_age = load_model(model_age_path)

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

# def predict_gender(img_cropped):
#     conf = model_gender.predict(img_cropped)[0]
#     # get label with max accuracy
#     idx = np.argmax(conf)
#     label = gender_classes[idx]
#     label = "{}: {:.2f}%".format(label, conf[idx] * 100)
#     return label


def predict_age(img_cropped):
    conf = model_age.predict(img_cropped)[0]
    # get label with max accuracy
    idx = np.argmax(conf)
    label = age_classes[idx]
    label = "{}: {:.2f}%".format(label, conf[idx] * 100)
    return label


def tra_predict_age(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Converting the coloured image to a grayscale image.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Converting the grayscale image to a canny edges filtered image.
    img = cv2.Canny(img,80,180)
    # Using the function defined above, extracting the features (mean and stdev values of all 10x10 pixel sections from the image) from the canny edges filtered image.
    img_features = features_grid(img)
    img_features = img_features.reshape(1, img_features.shape[0])
    idx = model_age.predict(img_features)[0]
    label = age_classes[idx]
    label = "{}: {:.2f}%".format(label, 0.99)
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
        # tra_face_crop = cv2.resize(face_crop, (200,200))
        age_face_crop = cv2.resize(face_crop, (200,200))
        # Converting the coloured image to a grayscale image.
        age_face_crop = cv2.cvtColor(age_face_crop, cv2.COLOR_BGR2GRAY)
        img_age_array = img_to_array(age_face_crop)
        img_age_array = np.expand_dims(img_age_array, axis=0)

        Y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(frame, "CNN (4 classes):" + predict_age(img_age_array), (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
    return frame
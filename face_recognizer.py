#!/usr/bin/python

# Import the required modules
import cv2, os
import types
import sys
import numpy as np
from PIL import Image
import ctypes

MB_OK = 0x0
MB_OKCXL = 0x01
MB_YESNOCXL = 0x03
MB_YESNO = 0x04
MB_HELP = 0x4000
ICON_EXLAIM=0x30
ICON_INFO = 0x40
ICON_STOP = 0x10

#Default classifier Name
filepath = './face_recognizer/Iknow.xml'

Message = 'User Found in Database'

Labels= ['Chris','Abid','Gred','Chang','Rick','Wong','Ramakant','Yang','Domnique','Sam','Abhishek','Jennie','Ted','Keshav','Ching','Rahul','Vishal']

# For face detection we will use the Haar Cascade provided by OpenCV.
cascadePath = "./face_recognizer/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# For face recognition we will the the LBPH Face Recognizer 
recognizer = cv2.face.createLBPHFaceRecognizer()

def get_images_and_labels(path):
    # Append all the absolute image paths in a list image_paths
    # We will not read the image with the .sad extension in the training set
    # Rather, we will use them to test our accuracy of the training
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.sad')]
    # images will contains face images
    images = []
    # labels will contains the label that is assigned to the image
    labels = []
    for image_path in image_paths:
        # Read the image and convert to grayscale
        image_pil = Image.open(image_path).convert('L')
        # Convert the image format into numpy array
        image = np.array(image_pil, 'uint8')
        # Get the label of the image
        nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        # Detect the face in the image
        faces = faceCascade.detectMultiScale(image)
        # If face is detected, append the face to images and the label to labels
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(nbr)
            cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
            cv2.waitKey(50)
    # return the images list and labels list
    return images, labels

# Path to the Yale Dataset
path = './face_recognizer/yalefaces'
path_Test = './face_recognizer/TestData'

# Added by Dinesh
#main function added , will get loaded once.
#will train only once , if classifier exist , it will not train
def main(arg1):
     #check if classifier is already trained.
    if arg1=="":
        # Call the get_images_and_labels function and get the face images and the 
        # corresponding labels
        images, labels = get_images_and_labels(path)
        cv2.destroyAllWindows()
   
        # Perform the tranining
        recognizer.train(images, np.array(labels))
        # save the Recognizer model to file. Dinesh
        recognizer.save(filepath)
    else:
        path = path_Test        
        predict_image()


def predict_image():
    # Append the images with the extension .sad into image_paths
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.sad')]
        #Load the classifier 
    recognizer.load(filepath)
    for image_path in image_paths:
        predict_image_pil = Image.open(image_path).convert('L')
        predict_image = np.array(predict_image_pil, 'uint8')
        faces = faceCascade.detectMultiScale(predict_image)
        for (x, y, w, h) in faces:
            nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
            Predicted = Labels[nbr_predicted]
            #commenting it as not required
            #nbr_actual = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
            #if nbr_actual == nbr_predicted:
            if conf < 10:
                Message = "%s" "%s" "%s" "%s" % ('Test Image of',Predicted, 'is Correctly Recognized with confidence',conf)
                break
            else:
                Message = "%s" "%s" "%s" "%s" % ('Test Image has low confidence score as', conf ,'and is wrongly Recognized as', Predicted)
                
            
            cv2.imshow("Recognizing Face", predict_image[y: y + h, x: x + w])
            cv2.waitKey(1000)
     
    ctypes.windll.user32.MessageBoxA(0, Message, "User Found ? ", MB_HELP| MB_YESNO | ICON_STOP)
     
########
#first argument is the path for the image to be predicted.
#if __name__ == '__main__':
#    Pred_image=''
#    if len(sys.argv)>0:
#        if sys.argv[1:]:
#            Pred_image =sys.argv[1]
#    
#    #for Testing.. 
#    Pred_image ="shot_0_001.sad"
#    main(Pred_image)
 
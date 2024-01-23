import pickle

import numpy as np
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
#from sklearn.neighbors import KNeighborsClassifier
import cv2


#STEP 2 - Dumping feature and filenames created
feature_list = pickle.load(open('embedding.pkl','rb'))
filenames = pickle.load(open('filename.pkl','rb'))

#STEP 3 - Model creation
model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])


#STEP 4 TEST Image feature extraction function and getting norm result
#(first we used we used this function to train our dataset now we'll train our uploaded img
# same parameter good accuracy)

img = image.load_img('uploads/black tee.jpg',target_size=(224,224))
img_arrey = image.img_to_array(img)
expanded_img_arrey = np.expand_dims(img_arrey,axis=0)
preprocessed_img = preprocess_input(expanded_img_arrey)
result = model.predict(preprocessed_img).flatten()
normalized_result = result/norm(result)

# STEP 4 - Distance calcuation matrics -- we have both extracted data set and trained and coverted test img
#BBOOM les GOO!

#Distance & indices milna hai

neighbors = NearestNeighbors(n_neighbors=6, metric='euclidean', algorithm ='brute')  #AWS me ANNOY use karenge
neighbors.fit(feature_list)   #OUR WHOLE DATA EXTRACTED FEATURE IS IN THIS LIST (VARIABLE OF PKL)

distance, indices = neighbors.kneighbors([normalized_result]) # distance milage aur neighbors ke indices milenge

#print(indices)

# Joining the indices that we'have got with filename (indices se filename call)

#for file in indices [0]: agar hum indices 0 se kall karenge to upar 6 tak difine hai wo aayega lekin
# 1st output khud he apna input rahega to slicing minor change in code

for file in indices[0][1:6]:
    tem_test = cv2.imread(filenames[file])
    cv2.imshow("output", cv2.resize(tem_test,(512,512)))
    cv2.waitKey(0)
 # NOW INSERT IMAGE IN TO PREPROCESS FUNNEL AND SEE RECOMMEDATION
 # ALSO DON'T FORGET TO PKL DUM AND PREPROCESS IN APP.PY
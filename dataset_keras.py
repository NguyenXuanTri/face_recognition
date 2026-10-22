import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
from os.path import join, isfile
from keras_facenet import FaceNet
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


from mtcnn.mtcnn import MTCNN

class FACELOADING:
    def __init__(self, directory):
        self.directory = directory           #luu duong dan D:\face_test\dataset
        self.X = []                          #tao mang X
        self.Y = []                          #tao mang Y
        self.detector = MTCNN()              #khoi tao bo phat hien khuon mat

    def extract_face(self, filename):
        img = cv.imread(filename)                                #doc anh tu file
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)                 #chuyen doi mau anh BGR -> RGB

        #x, y: toa do goc tren ben trai khuon mat
        #w, h: chieu rong va chieu cao vung khuon mat
        x, y, w, h = self.detector.detect_faces(img)[0]['box']   

        x,y = abs(x), abs(y)                                     #x, y khong am 
        face = img[y:y+h, x:x+w]                                 #cat vung khuon mat
        face_arr = cv.resize(face, (160, 160))                   #dua anh ve ti le 160x160 pixel 
        return face_arr                                          #tra ve anh sau khi xu ly
    
    def load_faces(self, dir):
        FACES = []                                               #mang 1 chieu chua anh khuon mat

        #duyet qua cac thu muc trong ronaldo/messi/rose
        for im_name in os.listdir(dir):                          
            try:
                #ghep duong dan -> D:\face_test\dataset\(ronaldo, messi, rose) + \x.jpg
                path = join(dir, im_name)             

                #xu ly anh tu duong dan sau do luu vao single_face
                single_face = self.extract_face(path)  

                #luu anh da xu ly vao mang FACES
                FACES.append(single_face)
            except Exception as e:
                pass                                              #neu co loi thi bo qua anh tiep

        #tra ve mang tat ca anh khuon mat da trich xuat thanh cong
        return FACES                                              
    
    def load_classes(self):
        for sub_dir in os.listdir(self.directory):                #duyet qua cac thu muc trong dataset

            #ghep duong dan -> D:\face_test\dataset + \(ronaldo, messi, rose)
            path = join(self.directory, sub_dir)

            #tat ca anh khuon mat da trich xuat thanh cong luu vao mang FACES
            FACES = self.load_faces(path)

            #dat ten theo sub_dir(ronaldo/messi) voi mang anh trong FACES roi luu vao labels
            #labels = ["ronaldo", "ronaldo", "ronaldo", ...]
            labels = [sub_dir for _ in range(len(FACES))]

            #in ra xem so anh load duoc co bang so anh trong dataset hay khong
            print(f'Loaded successfully: {len(labels)}')     

            self.X.extend(FACES)        #noi danh sach FACES(ronaldo-messi-rose) vao mang self.X
            self.Y.extend(labels)       #noi danh sach labels(ronaldo-messi-rose) tuong ung vao mang self.Y

        #chuyen danh sach python thanh numpy (numpy nhanh hon va ML yeu cau dang numpy)
        return np.asarray(self.X), np.asarray(self.Y)

faceloading = FACELOADING(r'D:\face_test\test')

#X: mang anh khuon mat
#Y: mang ten tuong ung
X, Y = faceloading.load_classes()

#khoi tao bo vector dac trung
embedder = FaceNet()   

def get_embedding(face_img):                       #nhan vao 1 anh da crop va resize
    face_img = face_img.astype('float32')          #chuyen kieu du lieu tu uint8 -> float32

    #chuyen anh (160, 160, 3) -> (1, 160, 160, 3)
    #1 anh -> 1 batch (lo) chua 1 anh
    face_img = np.expand_dims(face_img, axis=0)    

    yhat = embedder.embeddings(face_img)           #trich xuat vector dac trung

    #yhat = [[128 vector]] <=> (1, 128)
    #vi xu ly 1 batch anh nen se tra lai 1 batch 128 vector
    #yhat[0] = [128 vector] <=> (128) => bo di phan batch, lay phan 128 vector
    return yhat[0]  

EMBEDDED_X = []                                #tao mang EMBEDDED_X

for img in X:
    EMBEDDED_X.append(get_embedding(img))      #luu tung vector dac trung cua tung anh vao mang

EMBEDDED_X = np.asarray(EMBEDDED_X)            #chuyen danh sach tu dang Python -> Numpy

#luu tat ca vao file faces_embeddings.npz chua mang vector (arr_0) va mang labels (arr_1)
np.savez_compressed('faces_tests.npz', EMBEDDED_X, Y)

from PIL import Image
import matplotlib.pyplot as plt

num_images = 48
cols = 4
rows = num_images // cols

plt.figure(figsize=(10, 6))

for i in range(num_images):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(X[i].astype('uint8'))
    plt.axis('off')

plt.tight_layout()
plt.show()
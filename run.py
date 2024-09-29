from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import numpy as np 
import matplotlib.pyplot as plt
import cv2
from keras.models import load_model

model = load_model('face_recognition_model.h5')
#đường dẫn tới ảnh cần nhận diện
img_path = './Tony Stark/a.jpg'

#load ảnh sử dụng opencv

img_cv = cv2.imread(img_path)
img_cv_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

#TIỀN XỬ LÝ ẢNH VÀ ĐƯA VỀ DẠNG MẢNG
img = image.load_img(img_path, target_size = (224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)


#Dự đoán với mô hình đã huấn luyện   
prediction = model.predict(x)

# #lấy index của lớp với giá trị xác suất cao nhất
predicted_class = np.argmax(prediction[0], axis =-1)

#giả sử, bạn có một list hoặc dictionary ánh xạ giữa index và tên người 

label_mapping = {0:'Tony Stark',
                 1:'Elon Musk',
                 2:'Tom Cruise',
                 3:'Donal Trump',
                 4:'Justin Bieber',
                 5:'Shawn Mendes',}
predicted_name = label_mapping[predicted_class]


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascades/haarcascade_frontalface_alt.xml')
faces = face_cascade.detectMultiScale(img_cv_rgb, 1.3, 5)
fig, ax = plt.subplots(figsize=(10, 10))
for (x, y, w, h) in faces: 
    # vẽ hộp giới hạn xung quanh khuôn mặt 
    cv2.rectangle(img_cv_rgb, (x,y), (x + m, y + h), (255, 0 , 0), 3)

    #tiền xử lý ảnh cho việc dự đoán nhận diện
    face_img = img_cv_rgb[y:y+h, x:x+w] #cắt khuôn mặt từ ảnh 
    face_img = cv2.resize(face_img, (224, 224)) # thay đổi kích thước ảnh thành 224 x 224
    face_array = np.expand_dims(face_array, axis=0)
    face_array = preprocess_input(face_array) #Tiền xử lý ảnh
    
    
    #Dự đoán với mô hình đã huấn luyện 
    prediction = model.predict(face_array)
    predicted_class = np.argmax(prediction[0], axis = -1)
    predicted_name = label_mapping[predicted_class]
    
    #thêm tên vào phần hiển thị 
    cv2.putText(img_cv_rgb, predicted_name, (x,y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (36, 255, 12), 2)

plt.imshow(img_cv_rgb)
plt.axis('off')
plt.show()    
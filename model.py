import os
import numpy as np
import cv2
from keras.applications import MobileNetV3Small
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
import math
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import matplotlib.pyplot as plt
from keras.models import load_model



from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Load MobileNetV3 as a base model
base_model = MobileNetV3Small(weights='imagenet', include_top=False)

#Thêm các layer custom cho nhận diện khuôn mặt 
x = base_model.output
x = GlobalAveragePooling2D()(x)

#Thêm một fully-connected layer
x = Dense(1024, activation ='relu')(x)

num_classes = 6 # số lượng người cần nhận diện 
# Và một logistic layer -- ví dụ với 2 lớp (người này hoặc người kia)
predictions = Dense(num_classes, activation = 'softmax')(x)

model = Model(inputs = base_model.input, outputs = predictions)

#Đóng băng các convolutional base layer
for layer in base_model.layers:
    layer.trainable = False
    
   
    
#Compile mô hình (nên sử dụng mất mát'categorical_crossentropy')
model.compile(optimizer='rmsprop', loss = 'categorical_crossentropy')

#Chuẩn bị data augmentation configuration 
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)    

test_datagen = ImageDataGenerator(rescale=1./255)


predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# #nếu bạn mở lại toàn bộ layer để fine-tuning
# # bạn có thê set trainable = true cho từng layer hoặc sử dụng vòng lặp như sau: 
for layer in model.layers:
    layer.trainable = True
  
# #cập nhật thuộc tính validation_steps và steps_per_epoch nếu bạn biết số lượng ảnh của mình 
# # ví dụ, giả sử tập huấn luyên có 1200 ảnh và validation set có 300 ảnh, batch size là 32 
train_data_dir =  './train'
validation_data_dir ='./valid'
batch_size = 8
epochs = 30 # số vòng lặp huấn luyện
nb_train_samples = 148 # số lượng ảnh huấn luyện 
nb_validation_samples = 51  # số lượng ảnh validation

# # sử dụng data generator

train_generator = train_datagen.flow_from_directory(
    train_data_dir, 
    target_size = (224, 224),
    batch_size=batch_size, 
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

model.compile(optimizer = 'rmsprop',
              loss ='categorical_crossentropy',
              metrics =['accuracy'])
# Adjust steps per epoch
steps_per_epoch = math.ceil(nb_train_samples // batch_size)
validation_steps = math.ceil(nb_validation_samples // batch_size)

model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data = validation_generator,
    validation_steps=validation_steps
)
model.save('face_recognition_model.h5')  # Save your model
print(train_generator.class_indices)



img_path = './Tom Cruise/4.jpg'

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

predicted_class = np.argmax(prediction[0], axis =-1)

#giả sử, bạn có một list hoặc dictionary ánh xạ giữa index và tên người 

label_mapping = {0:'Tony Stark',
                 1:'Elon Musk',
                 2:'Tom Cruise',
                 3:'Donal Trump',
                 4:'Justin Bieber',
                 5:'Shawn Mendes',}
predicted_name = label_mapping[predicted_class]


face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
faces = face_cascade.detectMultiScale(img_cv_rgb, 1.3, 5)
fig, ax = plt.subplots(figsize=(10, 10))
for (x, y, w, h) in faces: 
    # vẽ hộp giới hạn xung quanh khuôn mặt 
    cv2.rectangle(img_cv_rgb, (x,y), (x + w, y + h), (255, 0 , 0), 3)

    #tiền xử lý ảnh cho việc dự đoán nhận diện
    face_img = img_cv_rgb[y:y+h, x:x+w] #cắt khuôn mặt từ ảnh 
    face_img = cv2.resize(face_img, (224, 224)) # thay đổi kích thước ảnh thành 224 x 224
    face_array = image.img_to_array(face_img) #chuyeenr thanh mang
    face_array = np.expand_dims(face_array, axis=0)
    face_array = preprocess_input(face_array) #Tiền xử lý ảnh
    
    
    #Dự đoán với mô hình đã huấn luyện 
    prediction = model.predict(face_array)
    predicted_class = np.argmax(prediction[0], axis = -1)
    predicted_name = label_mapping[predicted_class]
    
    #thêm tên vào phần hiển thị 
    cv2.putText(img_cv_rgb, predicted_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (36, 255, 12), 2)

plt.imshow(img_cv_rgb)
plt.axis('off')
plt.show()    
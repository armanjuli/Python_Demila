#proses    
#pelatihan cnn dan jst- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def Proses(self):
    import pandas as pd
    import cv2
from keras.preprocessing.image import ImageDataGenerator  #from tensorflow.keras_preprocessing.image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from PIL import Image
 
# Lokasi dataset sesuai dengan struktur folder
#csv_train_data_dir = 'train.csv' #train mengubah input menjadi output
#csv_validation_data_dir = 'validation.csv' #val mengukur akurasi model data "train" yg dimiliki (bisa berulang kali dilakukan)
#csv_test_data_dir = 'test.csv' #test untuk mengetahui akurasi test pada akurasi yang dimiliki sebelumnya dri data val
path_to_dataset = ('train.csv')
path_to_dataset = ('validation.csv')
path_to_dataset = ('test.csv')

dataset = pd.read_csv(path_to_dataset)
print(dataset.head())
# Data augmentation untuk data pelatihan
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
# Rescale data validasi dan pengujian
validation_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)
# Ukuran batch dan dimensi gambar
batch_size = 32
image_width, image_height = 150, 150
# Memuat data pelatihan, validasi, dan pengujian dari folder
train_generator = train_datagen.flow_from_directory(
    path_to_dataset,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode="categorical"
)
validation_generator = validation_datagen.flow_from_directory(
    path_to_dataset,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode='categorical'
)
test_generator = test_datagen.flow_from_directory(
    path_to_dataset,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# Membuat model CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_width, image_height, 3)),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # 3 kelas untuk kematangan buah nanas
])
# Kompilasi model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Melatih model
epochs = 10
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)
# Prediksi pada contoh gambar baru
# Replace 'path_to_new_image' with the actual path to the new image
new_image = tf.keras.preprocessing.image.load_img('path_to_new_image', target_size=(image_width, image_height))
new_image = tf.keras.preprocessing.image.img_to_array(new_image)
new_image = new_image.reshape((1,) + new_image.shape)
new_image /= 255.0

prediction = model.predict(new_image)
print(f'Prediction: {prediction}')
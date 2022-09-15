import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from sklearn.model_selection import train_test_split

model = tf.keras.models.load_model('./CIFAR_v2')

filepaths =  'data/test/1class5/img12.png'

IMAGE_SIZE  = (224, 224)
test_image = tf.keras.utils.load_img(filepaths
                            ,target_size =IMAGE_SIZE )
test_image = tf.keras.utils.img_to_array(test_image)
test_image = test_image.reshape((1, test_image.shape[0], test_image.shape[1], test_image.shape[2]))
test_image = preprocess_input(test_image)
prediction = model.predict(test_image)

df = pd.DataFrame({'pred':prediction[0]})
df = df.sort_values(by='pred', ascending=False, na_position='first')
print(f"## 예측률 : {(df.iloc[0]['pred'])* 100:.2f}%")
print([df[df == df.iloc[0]].index[0]])
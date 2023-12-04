import sklearn.utils
import tensorflow as tf
import numpy as np
import pandas as pd
import albumentations as A
import matplotlib.pyplot as plt
import os
import cv2
import gc
import absl.logging

from keras.applications.densenet import layers
from tensorflow.keras.layers import Input, Dense, Conv2D, Dropout, Flatten, Activation, MaxPooling2D, \
    GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.applications.efficientnet import preprocess_input as efnet_preprocess_input
from tensorflow.keras.applications.mobilenet import preprocess_input as mobile_preprocess_input
from importlib import reload
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# 초기 설정값
reload(tf.keras.models)
gc.collect()

# 들어오는 이미지 크기 설정
IMAGE_SIZE = 224
IMAGE_SIZEB1 = 240
IMAGE_SIZEB2 = 224
INPUT_SIZEB3 = 300
# 배치 사이즈 16 ~ 32
BATCH_SIZE = 16
# 반복 횟수
N_EPOCHS = 50
FIRST_EPOCHS = 5

# warning 방지용
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
absl.logging.set_verbosity(absl.logging.ERROR)


# test_set 하기
def show_history(history):
    plt.figure(figsize=(6, 6))
    plt.yticks(np.arange(0, 1, 0.05))
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='valid')
    plt.legend()
    plt.show()


#dataframe 생성 함수
def make_dataframe():
    paths = []
    label_gubuns = []
    # test용 이 따로 있으면 사용
    dataset_gubuns = []

    # 라벨 추가 작업(파일을 통해 라벨 만들기)
    for dirname, _, filenames in os.walk('dataSet/SkinDataSet/'):
        for filename in filenames:
            # 이미지 파일이 아닌 파일도 해당 디렉토리에 있음.
            if '.jpg' in filename:
                # 파일의 절대 경로를 file_path 변수에 할당.
                file_path = dirname + '/' + filename
                paths.append(file_path)
                # 파일의 절대 경로에 training_set, test_set가 포함되어 있으면 데이터 세트 구분을 'train'과 'test_set'로 분류.
                if 'training_set' in file_path:
                    dataset_gubuns.append('train')
                elif 'test_set' in file_path:
                    dataset_gubuns.append('test_set')
                else:
                    dataset_gubuns.append('N/A')

                if 'Atopic' in file_path:
                    label_gubuns.append('Atopic')
                elif 'Eczema' in file_path:
                    label_gubuns.append('Eczema')
                elif 'Normal' in file_path:
                    label_gubuns.append('Normal')
                else:
                    label_gubuns.append('N/A')

    data_df = pd.DataFrame({'path': paths, 'dataset': dataset_gubuns, 'label': label_gubuns})
    return data_df


# sequence기반 데이터셋 만들기
class Skin_Dataset(Sequence):
    def __init__(self, image_filenames, labels, batch_size=BATCH_SIZE, augmentor=None, shuffle=False, pre_func=None):
        self.shuffle = shuffle
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.augmentor = augmentor
        self.pre_func = pre_func
        if self.shuffle:
            self.on_epoch_end()
            pass

    def __len__(self):
        return int(np.ceil(len(self.labels) / self.batch_size))

    def __getitem__(self, index):
        image_name_batch = self.image_filenames[index * self.batch_size:(index + 1) * self.batch_size]
        if self.labels is not None:
            label_batch = self.labels[index * self.batch_size:(index + 1) * self.batch_size]

            image_batch = np.zeros((image_name_batch.shape[0], IMAGE_SIZEB2, IMAGE_SIZEB2, 3))
            for image_index in range(image_name_batch.shape[0]):
                image = cv2.cvtColor(cv2.imread(image_name_batch[image_index]), cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (IMAGE_SIZEB2, IMAGE_SIZEB2))
                if self.augmentor is not None:
                    image = self.augmentor(image=image)['image']

                    # 만일 preprocessing_input이 pre_func인자로 들어오면 이를 이용하여 scaling 적용.
                if self.pre_func is not None:
                    image = self.pre_func(image)

                image_batch[image_index] = image

            return image_batch, label_batch

    def on_epoch_end(self):
        if self.shuffle:
            self.image_filenames, self.labels = sklearn.utils.shuffle(self.image_filenames, self.labels)
        else:
            pass


# 사용할 모델 만들기
def create_model(model_name='efficientnetB0', verbose=False):
    input_tensor = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    if model_name == 'xception':
        base_model = tf.keras.applications.xception.Xception(
            include_top=False,
            weights='imagenet',
            input_tensor=input_tensor,
            pooling=None,
            classes=3,
            classifier_activation='softmax',
        )
    elif model_name == 'efficientnetB0':
        base_model = tf.keras.applications.efficientnet.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_tensor=input_tensor,
            pooling=None,
            classes=3,
            classifier_activation='softmax',
        )
    elif model_name == 'efficientnetB2':
        base_model = tf.keras.applications.efficientnet.EfficientNetB2(
            include_top=False,
            weights='imagenet',
            input_tensor=input_tensor,
            pooling=None,
            classes=3,
            classifier_activation='softmax',
        )

    bm_output = base_model.output

    x = GlobalAveragePooling2D()(bm_output)
    x = Dropout(rate=0.3)(x)
    x = Dense(300, activation='relu', name='fc1')(x)
    x = Dropout(rate=0.3)(x)
    output = Dense(3, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)

    if verbose == 0:
        model.summary()
    return model

#데이터프레임 생성
data_df = make_dataframe()

print(data_df['dataset'].value_counts())
print(data_df['label'].value_counts())

# 이미지 확인용
# Atopic_image_list = data_df[data_df['label'] == 'Atopic']['path'].iloc[:6].tolist()
# show_grid_images(Atopic_image_list, ncols=6, title='Atopic')

# Eczema_image_list = data_df[data_df['label'] == 'Eczema']['path'].iloc[:6].tolist()
# show_grid_images(Eczema_image_list, ncols=6, title='Eczema')

# Normal_image_list = data_df[data_df['label'] == 'Normal']['path'].iloc[:6].tolist()
# show_grid_images(Normal_image_list, ncols=6, title='Normal')

# dataFrame 생성하기
train_df = data_df[data_df['dataset'] == 'train']
test_df = data_df[data_df['dataset'] == 'test_set']

# 학습 데이터의 image path와 label을 Numpy array로 변환 및 Label encoding
train_path = train_df['path'].values
train_label = pd.get_dummies(train_df['label']).values
test_path = test_df['path'].values
test_label = pd.get_dummies(test_df['label']).values

tr_path, val_path, tr_label, val_label = train_test_split(train_path, train_label, test_size=0.15, random_state=2021)

#알규멘트 설정하기
skin_augmentor = A.Compose(
    [A.HorizontalFlip(p=0.5),
     A.VerticalFlip(p=0.5),
     A.ShiftScaleRotate(p=0.5)]
)

tr_ds = Skin_Dataset(tr_path, tr_label, batch_size=BATCH_SIZE, augmentor=skin_augmentor,
                     shuffle=True, pre_func=efnet_preprocess_input)
val_ds = Skin_Dataset(val_path, val_label, batch_size=BATCH_SIZE, augmentor=None,
                      shuffle=False, pre_func=efnet_preprocess_input)
test_ds = Skin_Dataset(test_path, test_label, batch_size=BATCH_SIZE, augmentor=None,
                       shuffle=False, pre_func=efnet_preprocess_input)

tr_image_batch = next(iter(tr_ds))[0]
tr_label_batch = next(iter(tr_ds))[1]
val_image_batch = next(iter(val_ds))[0]
val_label_batch = next(iter(val_ds))[1]
test_image_batch = next(iter(test_ds))[0]
test_label_batch = next(iter(test_ds))[1]

#이미지 스케일링 여부 확인기
#print(tr_image_batch[:1])
#print(val_image_batch[:1])

#모델 생성하기
model = create_model('efficientnetB2', 1)

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
# 3번 iteration내에 validation loss가 향상되지 않으면 learning rate을 기존 learning rate * 0.2로 줄임.
rlr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, mode='min', verbose=1)
# 10번 iteration내에 validation loss가 향상되지 않으면 더 이상 학습하지 않고 종료
ely_cb = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

#fine tuning 적용
for layer in model.layers[:-4]:
    layer.trainable = False
    # print(layer.name, 'trainable:', layer.trainable)

# 학습하기 1단계
history = model.fit(tr_ds, epochs=FIRST_EPOCHS,
                    validation_data=val_ds,
                    callbacks=[rlr_cb, ely_cb])

#학습하기 2단계

for layer in model.layers:
    if not isinstance(layer, layers.BatchNormalization):
        layer.trainable = True

model.compile(optimizer=Adam(learning_rate=0.0008), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(tr_ds, epochs=N_EPOCHS,
                    validation_data=val_ds,
                    callbacks=[rlr_cb, ely_cb])


# evaluation으로 성능 검증
show_history(history)
evaluation_result= model.evaluate(test_ds)
print(evaluation_result)


# Save the model.
# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()


with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

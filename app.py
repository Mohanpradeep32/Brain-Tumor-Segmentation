import os
import nibabel as nib
import numpy as np
import cv2
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect ,flash, url_for
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    concatenate,
    BatchNormalization,
    Activation,
    add,
    multiply,
)
SEGMENT_CLASSES = {
    0: 'NOT tumor',
    1: 'NECROTIC/CORE',  # or NON-ENHANCING tumor CORE
    2: 'EDEMA',
    3: 'ENHANCING'  # original 4 -> converted into 3 later
}

IMG_SIZE = 128
VOLUME_SLICES = 100
VOLUME_START_AT = 22
cur_dir = os.path.abspath(os.getcwd())

def ma_block(x, n_filters):
    x = Conv2D(n_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(n_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    return x

def attention_block(x, g, n_filters):
    theta = Conv2D(n_filters, kernel_size=1)(x)
    phi = Conv2D(n_filters, kernel_size=1)(g)
    f = Activation('relu')(add([theta, phi]))
    f = Conv2D(1, kernel_size=1)(f)
    f = Activation('sigmoid')(f)
    x = multiply([x, f])
    return x
def mdmodel(input_shape, n_classes):
    inputs = Input(shape=input_shape)

    conv1 = ma_block(inputs, 32)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = ma_block(pool1, 64)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = ma_block(pool2, 128)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = ma_block(pool3, 256)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = ma_block(pool4, 512)

    up6 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same')(conv5)
    conv4_att = attention_block(conv4, up6, 256)
    merge6 = concatenate([conv4_att, up6], axis=3)
    conv6 = ma_block(merge6, 256)

    up7 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same')(conv6)
    conv3_att = attention_block(conv3, up7, 128)
    merge7 = concatenate([conv3_att, up7], axis=3)
    conv7 = ma_block(merge7, 128)

    up8 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same')(conv7)
    conv2_att = attention_block(conv2, up8, 64)
    merge8 = concatenate([conv2_att, up8], axis=3)
    conv8 = ma_block(merge8, 64)

    up9 = Conv2DTranspose(32, kernel_size=2, strides=2, padding='same')(conv8)
    conv1_att = attention_block(conv1, up9, 32)
    merge9 = concatenate([conv1_att, up9], axis=3)
    conv9 = ma_block(merge9, 32)

    outputs = Conv2D(n_classes, kernel_size=1, activation='softmax')(conv9)

    return tf.keras.Model(inputs, outputs)

def predict_tumors(flair_path, ce_path, dropdown, start_slice=60):
    flair = nib.load(flair_path).get_fdata()
    ce = nib.load(ce_path).get_fdata()

    X = np.empty((VOLUME_SLICES, IMG_SIZE, IMG_SIZE, 2))

    for j in range(VOLUME_SLICES):
        X[j, :, :, 0] = cv2.resize(flair[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
        X[j, :, :, 1] = cv2.resize(ce[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))

    pred = unet.predict(X / np.max(X), verbose=1)

    threshold = 0.5  # Adjust the threshold value as needed

    if dropdown == SEGMENT_CLASSES[1]:
        edema = pred[:, :, :, 2]
        tumor_exists = np.any(edema[start_slice, :, :] > threshold)
        plt.imshow(edema[start_slice, :, :], cmap="OrRd", interpolation='none', alpha=0.3)
        plt.title(f'{SEGMENT_CLASSES[1]} predicted')
        plt.axis(False)

    elif dropdown == SEGMENT_CLASSES[2]:
        core = pred[:, :, :, 1]
        tumor_exists = np.any(core[start_slice, :, :] > threshold)
        plt.imshow(core[start_slice, :, :], cmap="OrRd", interpolation='none', alpha=0.3)
        plt.title(f'{SEGMENT_CLASSES[2]} predicted')
        plt.axis(False)

    elif dropdown == SEGMENT_CLASSES[3]:
        enhancing = pred[:, :, :, 3]
        tumor_exists = np.any(enhancing[start_slice, :, :] > threshold)
        plt.imshow(enhancing[start_slice, :, :], cmap="OrRd", interpolation='none', alpha=0.3)
        plt.title(f'{SEGMENT_CLASSES[3]} predicted')
        plt.axis(False)

    else:
        tumor_exists = False
        plt.imshow(pred[start_slice, :, :, 1:4], cmap="Reds", interpolation='none', alpha=0.3)
        plt.title('all predicted classes')
        plt.axis(False)

    plt.tight_layout()
    plt.savefig(cur_dir + '\\static\\out_img.jpg')

    if tumor_exists:
        return render_template('success.html', tumor_exists=True)
    else:
        return render_template('success.html', tumor_exists=False)




menuu = list(SEGMENT_CLASSES.values())[1:]

unet = mdmodel(input_shape=(IMG_SIZE, IMG_SIZE, 2), n_classes=len(SEGMENT_CLASSES))
unet.load_weights(r"C:\Users\MOHAN PRADEEP\Downloads\md_model_weights (1).h5")
print('\nready to Load....\n')
print(" ........")

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def home():
    return render_template("home.html", menuu=menuu)



@app.route('/outpg', methods=["GET", "POST"])
def outtputs():
    flair_path = os.path.abspath(request.form.get("flair_path")[1:-1])
    ce_path = os.path.abspath(request.form.get("ce_path")[1:-1])
    select_tumor = (request.form.get("part_of_tumor"))

    if not flair_path:
        return render_template("failure.html", message="missing flair path")
    if not ce_path:
        return render_template("failure.html", message="missing ce path")
    if select_tumor not in menuu and select_tumor != 'all_the_tumor':
        return render_template("failure.html", message="invalid tumor part")

    predict_tumors(flair_path=flair_path, ce_path=ce_path, dropdown=select_tumor)

    flair = nib.load(flair_path).get_fdata()
    plt.imshow(
        cv2.resize(flair[:, :, 60 + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)),
        cmap="gray", interpolation='none', alpha=1.0
    )
    plt.title('input image')
    plt.tight_layout()
    plt.savefig(cur_dir + '\\static\\input.jpg')

    return render_template(
        "success.html",
        flair_path="input.png",
        output_path="out_img.png"
    )

if __name__ == '__main__':
    app.run(debug=True)

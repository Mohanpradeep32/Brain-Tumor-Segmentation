import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, UpSampling2D,
                                     MaxPool2D, Concatenate, Dropout)


###### original unet ######


# import tensorflow as tf
# from tensorflow.keras.layers import (Conv2D, MaxPool2D, Input,
#                                      Cropping2D, Concatenate, Conv2DTranspose)


# def conv2x_pool(filters, pool=True):
#     def _conv_pool(x):
#         x = Conv2D(filters, activation='relu', padding='same',
#                    kernel_size=(3, 3))(x)
#         x_skip = x = Conv2D(filters, activation='relu', padding='same', kernel_size=3)(x)
#         if pool:
#             x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)
#             return x, x_skip
#         else:
#             return x
#     return _conv_pool


# def T_crop_concat(filters, crop_shape):
#     def _t_crop_concat(x, x_skip):
#         x = Conv2DTranspose(filters, kernel_size=(2, 2), padding='same',
#                             activation='relu', strides=2)(x)
#         # x_skip = Cropping2D(cropping=crop_shape)(x_skip)
#         return Concatenate(axis=-1)([x, x_skip])
#     return _t_crop_concat


# def Unet(img_shape=(299, 299, 1), classes=5):
#     x = inputs = Input(img_shape)

#     # contracting path (left side):
#     x, x0 = conv2x_pool(32)(x)
#     x, x1 = conv2x_pool(64)(x)
#     x, x2 = conv2x_pool(128)(x)
#     x, x3 = conv2x_pool(256)(x)

#     x = Conv2D(512, activation='relu',padding='same', kernel_size=(3, 3))(x)
#     x = Conv2D(512, activation='relu',padding='same', kernel_size=(3, 3))(x)

#     # expansive path (right side):
#     x = T_crop_concat(filters=256, crop_shape=(4, 4))(x, x3)
#     x = conv2x_pool(256, pool=False)(x)
#     x = T_crop_concat(filters=128, crop_shape=(16, 16))(x, x2)
#     x = conv2x_pool(128, pool=False)(x)
#     x = T_crop_concat(filters=64, crop_shape=(40, 40))(x, x1)
#     x = conv2x_pool(64, pool=False)(x)
#     x = T_crop_concat(filters=32, crop_shape=(88, 88))(x, x0)
#     x = conv2x_pool(32, pool=False)(x)

#     outputs = Conv2D(classes, kernel_size=1, padding='same', strides=1)(x)

#     return tf.keras.Model(inputs, outputs, name="U-Net")


##### tweeked unet #####


def U_net(input_shape, classes, dropout=0.2):

    x = inputs = Input(shape=input_shape)

    # encoder part:
    x = Conv2D(32, kernel_size=3, activation='relu', padding='same')(x)
    x = x1 = Conv2D(32, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPool2D(pool_size=2)(x)

    x = Conv2D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = x2 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPool2D(pool_size=2)(x)

    x = Conv2D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = x3 = Conv2D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPool2D(pool_size=2)(x)

    x = Conv2D(256, kernel_size=3, activation='relu', padding='same')(x)
    x = x4 = Conv2D(256, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPool2D(pool_size=2)(x)

    x = Conv2D(512, kernel_size=3, activation='relu', padding='same')(x)
    x = Conv2D(512, kernel_size=3, activation='relu', padding='same')(x)
    x = Dropout(dropout)(x)

    # decoder part:
    x = Conv2D(256, kernel_size=2, activation='relu',
               padding='same')(UpSampling2D(size=2)(x))
    x = Concatenate(axis=-1)([x4, x])
    x = Conv2D(256, 3, activation='relu', padding='same')(x)
    x = Conv2D(256, 3, activation='relu', padding='same')(x)

    x = Conv2D(128, kernel_size=2, activation='relu',
               padding='same')(UpSampling2D(size=2)(x))
    x = Concatenate(axis=-1)([x3, x])
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)

    x = Conv2D(64, kernel_size=2, activation='relu',
               padding='same')(UpSampling2D(size=2)(x))
    x = Concatenate(axis=-1)([x2, x])
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)

    x = Conv2D(32, kernel_size=2, activation='relu',
               padding='same')(UpSampling2D(size=2)(x))
    x = Concatenate(axis=-1)([x1, x])
    x = Conv2D(32, 3, activation='relu', padding='same')(x)
    x = Conv2D(32, 3, activation='relu', padding='same')(x)

    x = outputs = Conv2D(classes, kernel_size=1,
                         padding='same', activation='softmax')(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


###### unet++ / Nest Unet #####


# import tensorflow as tf
# from tensorflow.keras.layers import (Input, Conv2D, Dropout, MaxPool2D,
#                                      Conv2DTranspose, Concatenate)

# def nest_unet(img_shape=(128, 128, 2), classes=4):
#     nb_filter = [32, 64, 128, 256, 512]
#     # Build U-Net++ model
#     x = inputs = Input(shape=img_shape)

#     c1 = Conv2D(32, kernel_size=3, activation='elu',
#                 kernel_initializer='he_normal', padding='same')(x)
#     c1 = Dropout(0.5)(c1)
#     c1 = Conv2D(32, kernel_size=3, activation='elu',
#                 kernel_initializer='he_normal', padding='same')(c1)
#     c1 = Dropout(0.5)(c1)
#     p1 = MaxPool2D((2, 2), strides=(2, 2))(c1)

#     c2 = Conv2D(64, kernel_size=3, activation='elu',
#                 kernel_initializer='he_normal', padding='same')(p1)
#     c2 = Dropout(0.5)(c2)
#     c2 = Conv2D(64, kernel_size=3, activation='elu',
#                 kernel_initializer='he_normal', padding='same')(c2)
#     c2 = Dropout(0.5)(c2)
#     p2 = MaxPool2D((2, 2), strides=(2, 2))(c2)

#     up1_2 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(
#         2, 2), name='up12', padding='same')(c2)
#     conv1_2 = Concatenate([up1_2, c1], name='merge12', axis=3)
#     c3 = Conv2D(32, (3, 3), activation='elu',
#                 kernel_initializer='he_normal', padding='same')(conv1_2)
#     c3 = Dropout(0.5)(c3)
#     c3 = Conv2D(32, (3, 3), activation='elu',
#                 kernel_initializer='he_normal', padding='same')(c3)
#     c3 = Dropout(0.5)(c3)

#     conv3_1 = Conv2D(128, (3, 3), activation='elu',
#                     kernel_initializer='he_normal', padding='same')(p2)
#     conv3_1 = Dropout(0.5)(conv3_1)
#     conv3_1 = Conv2D(128, (3, 3), activation='elu',
#                     kernel_initializer='he_normal', padding='same')(conv3_1)
#     conv3_1 = Dropout(0.5)(conv3_1)
#     pool3 = MaxPool2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

#     up2_2 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(
#         2, 2), name='up22', padding='same')(conv3_1)
#     conv2_2 = Concatenate([up2_2, c2], name='merge22', axis=3)  # x10
#     conv2_2 = Conv2D(64, (3, 3), activation='elu',
#                     kernel_initializer='he_normal', padding='same')(conv2_2)
#     conv2_2 = Dropout(0.5)(conv2_2)
#     conv2_2 = Conv2D(64, (3, 3), activation='elu',
#                     kernel_initializer='he_normal', padding='same')(conv2_2)
#     conv2_2 = Dropout(0.5)(conv2_2)

#     up1_3 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(
#         2, 2), name='up13', padding='same')(conv2_2)
#     conv1_3 = Concatenate([up1_3, c1, c3], name='merge13', axis=3)
#     conv1_3 = Conv2D(32, (3, 3), activation='elu',
#                     kernel_initializer='he_normal', padding='same')(conv1_3)
#     conv1_3 = Dropout(0.5)(conv1_3)
#     conv1_3 = Conv2D(32, (3, 3), activation='elu',
#                     kernel_initializer='he_normal', padding='same')(conv1_3)
#     conv1_3 = Dropout(0.5)(conv1_3)

#     conv4_1 = Conv2D(256, (3, 3), activation='elu',
#                     kernel_initializer='he_normal', padding='same')(pool3)
#     conv4_1 = Dropout(0.5)(conv4_1)
#     conv4_1 = Conv2D(256, (3, 3), activation='elu',
#                     kernel_initializer='he_normal', padding='same')(conv4_1)
#     conv4_1 = Dropout(0.5)(conv4_1)
#     pool4 = MaxPool2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

#     up3_2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(
#         2, 2), name='up32', padding='same')(conv4_1)
#     conv3_2 = Concatenate([up3_2, conv3_1], name='merge32', axis=3)  # x20
#     conv3_2 = Conv2D(128, (3, 3), activation='elu',
#                     kernel_initializer='he_normal', padding='same')(conv3_2)
#     conv3_2 = Dropout(0.5)(conv3_2)
#     conv3_2 = Conv2D(128, (3, 3), activation='elu',
#                     kernel_initializer='he_normal', padding='same')(conv3_2)
#     conv3_2 = Dropout(0.5)(conv3_2)

#     up2_3 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(
#         2, 2), name='up23', padding='same')(conv3_2)
#     conv2_3 = Concatenate([up2_3, c2, conv2_2], name='merge23', axis=3)
#     conv2_3 = Conv2D(64, (3, 3), activation='elu',
#                     kernel_initializer='he_normal', padding='same')(conv2_3)
#     conv2_3 = Dropout(0.5)(conv2_3)
#     conv2_3 = Conv2D(64, (3, 3), activation='elu',
#                     kernel_initializer='he_normal', padding='same')(conv2_3)
#     conv2_3 = Dropout(0.5)(conv2_3)

#     up1_4 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(
#         2, 2), name='up14', padding='same')(conv2_3)
#     conv1_4 = Concatenate([up1_4, c1, c3, conv1_3], name='merge14', axis=3)
#     conv1_4 = Conv2D(32, (3, 3), activation='elu',
#                     kernel_initializer='he_normal', padding='same')(conv1_4)
#     conv1_4 = Dropout(0.5)(conv1_4)
#     conv1_4 = Conv2D(32, (3, 3), activation='elu',
#                     kernel_initializer='he_normal', padding='same')(conv1_4)
#     conv1_4 = Dropout(0.5)(conv1_4)

#     conv5_1 = Conv2D(512, (3, 3), activation='elu',
#                     kernel_initializer='he_normal', padding='same')(pool4)
#     conv5_1 = Dropout(0.5)(conv5_1)
#     conv5_1 = Conv2D(512, (3, 3), activation='elu',
#                     kernel_initializer='he_normal', padding='same')(conv5_1)
#     conv5_1 = Dropout(0.5)(conv5_1)

#     up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(
#         2, 2), name='up42', padding='same')(conv5_1)
#     conv4_2 = Concatenate([up4_2, conv4_1], name='merge42', axis=3)  # x30
#     conv4_2 = Conv2D(256, (3, 3), activation='elu',
#                     kernel_initializer='he_normal', padding='same')(conv4_2)
#     conv4_2 = Dropout(0.5)(conv4_2)
#     conv4_2 = Conv2D(256, (3, 3), activation='elu',
#                     kernel_initializer='he_normal', padding='same')(conv4_2)
#     conv4_2 = Dropout(0.5)(conv4_2)

#     up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(
#         2, 2), name='up33', padding='same')(conv4_2)
#     conv3_3 = Concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=3)
#     conv3_3 = Conv2D(128, (3, 3), activation='elu',
#                     kernel_initializer='he_normal', padding='same')(conv3_3)
#     conv3_3 = Dropout(0.5)(conv3_3)
#     conv3_3 = Conv2D(128, (3, 3), activation='elu',
#                     kernel_initializer='he_normal', padding='same')(conv3_3)
#     conv3_3 = Dropout(0.5)(conv3_3)

#     up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(
#         2, 2), name='up24', padding='same')(conv3_3)
#     conv2_4 = Concatenate([up2_4, c2, conv2_2, conv2_3], name='merge24', axis=3)
#     conv2_4 = Conv2D(64, (3, 3), activation='elu',
#                     kernel_initializer='he_normal', padding='same')(conv2_4)
#     conv2_4 = Dropout(0.5)(conv2_4)
#     conv2_4 = Conv2D(64, (3, 3), activation='elu',
#                     kernel_initializer='he_normal', padding='same')(conv2_4)
#     conv2_4 = Dropout(0.5)(conv2_4)

#     up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(
#         2, 2), name='up15', padding='same')(conv2_4)
#     conv1_5 = Concatenate([up1_5, c1, c3, conv1_3, conv1_4],
#                         name='merge15', axis=3)
#     conv1_5 = Conv2D(32, (3, 3), activation='elu',
#                     kernel_initializer='he_normal', padding='same')(conv1_5)
#     conv1_5 = Dropout(0.5)(conv1_5)
#     conv1_5 = Conv2D(32, (3, 3), activation='elu',
#                     kernel_initializer='he_normal', padding='same')(conv1_5)
#     conv1_5 = Dropout(0.5)(conv1_5)

#     outputs = nestnet_output_4 = Conv2D(classes, (1, 1), activation='softmax',
#                             kernel_initializer='he_normal',  name='output_4', padding='same')(conv1_5)

#     return tf.keras.Model(inputs, outputs)

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import (
    Input, Dense, Activation, Flatten,
    Conv2D, DepthwiseConv2D, SeparableConv2D,
    MaxPooling2D, AveragePooling2D,
    BatchNormalization, SpatialDropout2D, Dropout,
    Permute
)


def EEGNet(nb_classes, Chans=64, Samples=128,
           dropoutRate=0.5, kernLength=64,
           F1=8, D=2, F2=16, norm_rate=0.25,
           dropoutType='Dropout'):
    """Keras implementation of EEGNet

    Arguments:
        nb_classes (int): number of classes to classify
        Chans (int): number of channels in the EEG data
        Samples (int): number of time points in the EEG data
        dropoutRate (float): dropout fraction
        kernLength (int): length of temporal convolution in first layer.
                        Recommended to be set half the sampling rate.
        F1 (int): number of temporal filters to learn. Default: F1 = 8.
        F2 (int): number of pointwise filters to learn. Default: F2 = F1 * D.
        D (int): number of spatial filters to learn within each temporal
                        convolution. Default: D = 2.
        dropoutType (str): Either SpatialDropout2D or Dropout.

    Returns:
        Model: Keras Model instance
    """

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(Chans, Samples, 1))

    block1 = Conv2D(F1, (1, kernLength), padding='same',
                    input_shape=(Chans, Samples, 1),
                    use_bias=False)(input1)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropoutType(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, 16), use_bias=False,
                             padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = dropoutType(dropoutRate)(block2)

    flatten = Flatten(name='flatten')(block2)

    dense = Dense(1, name='dense',
                  kernel_constraint=max_norm(norm_rate))(flatten)
    sigmoid = Activation('sigmoid', name='sigmoid')(dense)

    return Model(inputs=input1, outputs=sigmoid)
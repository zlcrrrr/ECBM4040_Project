from tensorflow.keras.layers import Input,BatchNormalization,Conv2D,MaxPooling2D,Activation,AveragePooling2D,Flatten,Dropout,Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model

from .units import residual_unit
from .units import attention_unit

def AttentionResNet56(shape=(32,32,3), in_channel=32, kernel_size=5, n_classes=10, regularization=0.01):

    """
    shape: The shape of input data for one sample. (The shape of all input data is: (batch size, 32, 32, 3))
    in_channel: The 4-th dimension (channel number) of input weight matrix. For example, in_channel=3 means the input contains 3 channels.
    :param kernel_size: Integer. the shape of the kernel. For example, default kernel_size = 3 means you have a 3*3 kernel.
    :param n_classes: Integer. The number of target classes. For example, n_classes = 10 means you have 10 class labels.
    :param dropout: Float between 0 and 1. Fraction of the input units to drop.
    :param regularization: Float. Fraction of the input units to drop.
    """

    input_layer = Input(shape=(32,32,3))  # (batch size, 32, 32, 3)
    x = Conv2D(in_channel, kernel_size=kernel_size, padding='same')(input_layer)  # (batch size, 32, 32, 32)
    x = MaxPooling2D(pool_size=2)(x)  # (batch size, 16, 16, 32)

    # The 1st Attention Module
    x = residual_unit(x, input_channel=32, output_channel=128)  # (batch size, 16, 16, 128)
    x = attention_unit(x, skip=2) # (batch size, 16, 16, 128)

    # The 2nd Attention Module
    x = residual_unit(x, input_channel=128, output_channel=256, stride=2)  # (batch size, 8, 8, 256)
    x = attention_unit(x, skip=1) # (batch size, 8, 8, 256)

    # The 3rd Attention Module
    x = residual_unit(x, input_channel=256, output_channel=512, stride=2)  # (batch size, 4, 4, 512)
    x = attention_unit(x, skip=0) # (batch size, 4, 4, 512)
    
    x = residual_unit(x, input_channel=512, output_channel=1024)   # (batch size, 4, 4, 1024)
    x = residual_unit(x, input_channel=1024, output_channel=1024)  # (batch size, 4, 4, 1024)
    x = residual_unit(x, input_channel=1024, output_channel=1024)  # (batch size, 4, 4, 1024)
    
    x = Activation('relu')(x)

    x = AveragePooling2D(pool_size=4, strides=1)(x)  # (batch size, 1, 1, 1024)
    x = Flatten()(x) # (batch size, 1024)

    output = Dense(n_classes, kernel_regularizer=l2(regularization), activation='softmax')(x)

    model = Model(input_layer, output)

    return model



def AttentionResNet92(shape=(32,32,3), in_channel=32, kernel_size=5, n_classes=10, regularization=0.00, dropout=0.0):

    """
    shape: The shape of input data for one sample. (The shape of all input data is: (batch size, 32, 32, 3))
    in_channel: The 4-th dimension (channel number) of input weight matrix. For example, in_channel=3 means the input contains 3 channels.
    :param kernel_size: Integer. the shape of the kernel. For example, default kernel_size = 3 means you have a 3*3 kernel.
    :param n_classes: Integer. The number of target classes. For example, n_classes = 10 means you have 10 class labels.
    :param dropout: Float between 0 and 1. Fraction of the input units to drop.
    :param regularization: Float. Fraction of the input units to drop.
    """

    input_layer = Input(shape=(32,32,3))  # (batch size, 32, 32, 3)
    x = Conv2D(in_channel, kernel_size=kernel_size, padding='same')(input_layer)  # (batch size, 32, 32, 32)
    x = MaxPooling2D(pool_size=2)(x)  # (batch size, 16, 16, 32)

    # The 1st Attention Module
    x = residual_unit(x, input_channel=32, output_channel=128)  # (batch size, 16, 16, 128)
    x = attention_unit(x, skip=2) # (batch size, 16, 16, 128)
    
    # The 2nd Attention Module
    x = residual_unit(x, input_channel=128, output_channel=256, stride=2)  # (batch size, 8, 8, 256)
    x = attention_unit(x, skip=1) # (batch size, 8, 8, 256)
    x = attention_unit(x, skip=1) # (batch size, 8, 8, 256)

    # The 3rd Attention Module
    x = residual_unit(x, input_channel=256, output_channel=512, stride=2)  # (batch size, 4, 4, 512)
    x = attention_unit(x, skip=0) # (batch size, 4, 4, 512)
    x = attention_unit(x, skip=0) # (batch size, 4, 4, 512)
    x = attention_unit(x, skip=0) # (batch size, 4, 4, 512)

    
    x = residual_unit(x, input_channel=512, output_channel=1024)   # (batch size, 4, 4, 1024)
    x = residual_unit(x, input_channel=1024, output_channel=1024)  # (batch size, 4, 4, 1024)
    x = residual_unit(x, input_channel=1024, output_channel=1024)  # (batch size, 4, 4, 1024)
    
    # LZ add
#     x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = AveragePooling2D(pool_size=4, strides=1)(x)  # (batch size, 1, 1, 1024)
    x = Flatten()(x) # (batch size, 1024)
    
    x = Dropout(dropout)(x)
    output = Dense(n_classes, kernel_regularizer=l2(regularization), activation='softmax')(x)

    model = Model(input_layer, output)

    return model


def AttentionResNet128(shape=(32,32,3), in_channel=32, kernel_size=5, n_classes=10, regularization=0., dropout=0.0):

    """
    shape: The shape of input data for one sample. (The shape of all input data is: (batch size, 32, 32, 3))
    in_channel: The 4-th dimension (channel number) of input weight matrix. For example, in_channel=3 means the input contains 3 channels.
    :param kernel_size: Integer. the shape of the kernel. For example, default kernel_size = 3 means you have a 3*3 kernel.
    :param n_classes: Integer. The number of target classes. For example, n_classes = 10 means you have 10 class labels.
    :param dropout: Float between 0 and 1. Fraction of the input units to drop.
    :param regularization: Float. Fraction of the input units to drop.
    """

    input_layer = Input(shape=(32,32,3))  # (batch size, 32, 32, 3)
    x = Conv2D(in_channel, kernel_size=kernel_size, padding='same')(input_layer)  # (batch size, 32, 32, 32)
    x = MaxPooling2D(pool_size=2)(x)  # (batch size, 16, 16, 32)

    # The 1st Attention Module
    x = residual_unit(x, input_channel=32, output_channel=128)  # (batch size, 16, 16, 128)
    x = attention_unit(x, skip=2) # (batch size, 16, 16, 128)
    x = attention_unit(x, skip=2) # (batch size, 16, 16, 128)
    x = attention_unit(x, skip=2) # (batch size, 16, 16, 128)

    # The 2nd Attention Module
    x = residual_unit(x, input_channel=128, output_channel=256, stride=2)  # (batch size, 8, 8, 256)
    x = attention_unit(x, skip=1) # (batch size, 8, 8, 256)
    x = attention_unit(x, skip=1) # (batch size, 8, 8, 256)
    x = attention_unit(x, skip=1) # (batch size, 8, 8, 256)

    # The 3rd Attention Module
    x = residual_unit(x, input_channel=256, output_channel=512, stride=2)  # (batch size, 4, 4, 512)
    x = attention_unit(x, skip=0) # (batch size, 4, 4, 512)
    x = attention_unit(x, skip=0) # (batch size, 4, 4, 512)
    x = attention_unit(x, skip=0) # (batch size, 4, 4, 512)

    
    x = residual_unit(x, input_channel=512, output_channel=1024)   # (batch size, 4, 4, 1024)
    x = residual_unit(x, input_channel=1024, output_channel=1024)  # (batch size, 4, 4, 1024)
    x = residual_unit(x, input_channel=1024, output_channel=1024)  # (batch size, 4, 4, 1024)
    
    x = Activation('relu')(x)

    x = AveragePooling2D(pool_size=4, strides=1)(x)  # (batch size, 1, 1, 1024)
    x = Flatten()(x) # (batch size, 1024)

    output = Dense(n_classes, kernel_regularizer=l2(regularization), activation='softmax')(x)

    model = Model(input_layer, output)

    return model




def AttentionResNet164(shape=(32,32,3), in_channel=32, kernel_size=5, n_classes=10, regularization=0., dropout=0.0):

    """
    shape: The shape of input data for one sample. (The shape of all input data is: (batch size, 32, 32, 3))
    in_channel: The 4-th dimension (channel number) of input weight matrix. For example, in_channel=3 means the input contains 3 channels.
    :param kernel_size: Integer. the shape of the kernel. For example, default kernel_size = 3 means you have a 3*3 kernel.
    :param n_classes: Integer. The number of target classes. For example, n_classes = 10 means you have 10 class labels.
    :param dropout: Float between 0 and 1. Fraction of the input units to drop.
    :param regularization: Float. Fraction of the input units to drop.
    """

    input_layer = Input(shape=(32,32,3))  # (batch size, 32, 32, 3)
    x = Conv2D(in_channel, kernel_size=kernel_size, padding='same')(input_layer)  # (batch size, 32, 32, 32)
    x = MaxPooling2D(pool_size=2)(x)  # (batch size, 16, 16, 32)

    # The 1st Attention Module
    x = residual_unit(x, input_channel=32, output_channel=128)  # (batch size, 16, 16, 128)
    x = attention_unit(x, skip=3) # (batch size, 16, 16, 128)
    x = attention_unit(x, skip=3) # (batch size, 16, 16, 128)
    x = attention_unit(x, skip=3) # (batch size, 16, 16, 128)
    x = attention_unit(x, skip=3) # (batch size, 16, 16, 128)

    # The 2nd Attention Module
    x = residual_unit(x, input_channel=128, output_channel=256, stride=2)  # (batch size, 8, 8, 256)
    x = attention_unit(x, skip=2) # (batch size, 8, 8, 256)
    x = attention_unit(x, skip=2) # (batch size, 8, 8, 256)
    x = attention_unit(x, skip=2) # (batch size, 8, 8, 256)
    x = attention_unit(x, skip=2) # (batch size, 8, 8, 256)

    # The 3rd Attention Module
    x = residual_unit(x, input_channel=256, output_channel=512, stride=2)  # (batch size, 4, 4, 512)
    x = attention_unit(x, skip=1) # (batch size, 4, 4, 512)
    x = attention_unit(x, skip=1) # (batch size, 4, 4, 512)
    x = attention_unit(x, skip=1) # (batch size, 4, 4, 512)
    x = attention_unit(x, skip=1) # (batch size, 4, 4, 512)

    
    x = residual_unit(x, input_channel=512, output_channel=1024)   # (batch size, 4, 4, 1024)
    x = residual_unit(x, input_channel=1024, output_channel=1024)  # (batch size, 4, 4, 1024)
    x = residual_unit(x, input_channel=1024, output_channel=1024)  # (batch size, 4, 4, 1024)
    
    x = Activation('relu')(x)

    x = AveragePooling2D(pool_size=4, strides=1)(x)  # (batch size, 1, 1, 1024)
    x = Flatten()(x) # (batch size, 1024)

    output = Dense(n_classes, kernel_regularizer=l2(regularization), activation='softmax')(x)

    model = Model(input_layer, output)

    return model


def AttentionResNet236(shape=(32,32,3), in_channel=32, kernel_size=5, n_classes=10, regularization=0.00, dropout=0.0):

    """
    shape: The shape of input data for one sample. (The shape of all input data is: (batch size, 32, 32, 3))
    in_channel: The 4-th dimension (channel number) of input weight matrix. For example, in_channel=3 means the input contains 3 channels.
    :param kernel_size: Integer. the shape of the kernel. For example, default kernel_size = 3 means you have a 3*3 kernel.
    :param n_classes: Integer. The number of target classes. For example, n_classes = 10 means you have 10 class labels.
    :param dropout: Float between 0 and 1. Fraction of the input units to drop.
    :param regularization: Float. Fraction of the input units to drop.
    """

    input_layer = Input(shape=(32,32,3))  # (batch size, 32, 32, 3)
    x = Conv2D(in_channel, kernel_size=kernel_size, padding='same')(input_layer)  # (batch size, 32, 32, 32)
    x = MaxPooling2D(pool_size=2)(x)  # (batch size, 16, 16, 32)

    # The 1st Attention Module
    x = residual_unit(x, input_channel=32, output_channel=128)  # (batch size, 16, 16, 128)
    x = attention_unit(x, skip=2) # (batch size, 16, 16, 128)
    x = attention_unit(x, skip=2) # (batch size, 16, 16, 128)
    x = attention_unit(x, skip=2) # (batch size, 16, 16, 128)
    x = attention_unit(x, skip=2) # (batch size, 16, 16, 128)
    x = attention_unit(x, skip=2) # (batch size, 16, 16, 128)
    x = attention_unit(x, skip=2) # (batch size, 16, 16, 128)
    
    # The 2nd Attention Module
    x = residual_unit(x, input_channel=128, output_channel=256, stride=2)  # (batch size, 8, 8, 256)
    x = attention_unit(x, skip=1) # (batch size, 8, 8, 256)
    x = attention_unit(x, skip=1) # (batch size, 8, 8, 256)
    x = attention_unit(x, skip=1) # (batch size, 8, 8, 256)
    x = attention_unit(x, skip=1) # (batch size, 8, 8, 256)
    x = attention_unit(x, skip=1) # (batch size, 8, 8, 256)
    x = attention_unit(x, skip=1) # (batch size, 8, 8, 256)

    # The 3rd Attention Module
    x = residual_unit(x, input_channel=256, output_channel=512, stride=2)  # (batch size, 4, 4, 512)
    x = attention_unit(x, skip=0) # (batch size, 4, 4, 512)
    x = attention_unit(x, skip=0) # (batch size, 4, 4, 512)
    x = attention_unit(x, skip=0) # (batch size, 4, 4, 512)
    x = attention_unit(x, skip=0) # (batch size, 4, 4, 512)
    x = attention_unit(x, skip=0) # (batch size, 4, 4, 512)
    x = attention_unit(x, skip=0) # (batch size, 4, 4, 512)

    
    x = residual_unit(x, input_channel=512, output_channel=1024)   # (batch size, 4, 4, 1024)
    x = residual_unit(x, input_channel=1024, output_channel=1024)  # (batch size, 4, 4, 1024)
    x = residual_unit(x, input_channel=1024, output_channel=1024)  # (batch size, 4, 4, 1024)
    
    # LZ add
#     x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = AveragePooling2D(pool_size=4, strides=1)(x)  # (batch size, 1, 1, 1024)
    x = Flatten()(x) # (batch size, 1024)
    
    x = Dropout(dropout)(x)
    output = Dense(n_classes, kernel_regularizer=l2(regularization), activation='softmax')(x)

    model = Model(input_layer, output)

    return model


def AttentionResNet92_test(shape=(32,32,3), in_channel=16, kernel_size=3, n_classes=10, regularization=0.00, dropout=0.0):

    """
    shape: The shape of input data for one sample. (The shape of all input data is: (batch size, 32, 32, 3))
    in_channel: The 4-th dimension (channel number) of input weight matrix. For example, in_channel=3 means the input contains 3 channels.
    :param kernel_size: Integer. the shape of the kernel. For example, default kernel_size = 3 means you have a 3*3 kernel.
    :param n_classes: Integer. The number of target classes. For example, n_classes = 10 means you have 10 class labels.
    :param dropout: Float between 0 and 1. Fraction of the input units to drop.
    :param regularization: Float. Fraction of the input units to drop.
    """

    input_layer = Input(shape=(32,32,3))  # (batch size, 32, 32, 3)
    x = Conv2D(in_channel, kernel_size=kernel_size, padding='same')(input_layer)  # (batch size, 32, 32, 32)
    x = MaxPooling2D(pool_size=2)(x)  # (batch size, 16, 16, 32)

    # The 1st Attention Module
    x = residual_unit(x, input_channel=16, output_channel=128)  # (batch size, 16, 16, 128)
    x = attention_unit(x, skip=2) # (batch size, 16, 16, 128)
    
    # The 2nd Attention Module
    x = residual_unit(x, input_channel=128, output_channel=256, stride=2)  # (batch size, 8, 8, 256)
    x = attention_unit(x, skip=1) # (batch size, 8, 8, 256)
    x = attention_unit(x, skip=1) # (batch size, 8, 8, 256)

    # The 3rd Attention Module
    x = residual_unit(x, input_channel=256, output_channel=512, stride=2)  # (batch size, 4, 4, 512)
    x = attention_unit(x, skip=0) # (batch size, 4, 4, 512)
    x = attention_unit(x, skip=0) # (batch size, 4, 4, 512)
    x = attention_unit(x, skip=0) # (batch size, 4, 4, 512)

    
    x = residual_unit(x, input_channel=512, output_channel=1024)   # (batch size, 4, 4, 1024)
    x = residual_unit(x, input_channel=1024, output_channel=1024)  # (batch size, 4, 4, 1024)
    x = residual_unit(x, input_channel=1024, output_channel=1024)  # (batch size, 4, 4, 1024)
    
    # LZ add
#     x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = AveragePooling2D(pool_size=4, strides=1)(x)  # (batch size, 1, 1, 1024)
    x = Flatten()(x) # (batch size, 1024)
    
    x = Dropout(dropout)(x)
    output = Dense(n_classes, kernel_regularizer=l2(regularization), activation='softmax')(x)

    model = Model(input_layer, output)

    return model
from tensorflow.keras.layers import Input,BatchNormalization,Conv2D,MaxPooling2D,Activation,AveragePooling2D,Flatten,Dropout,Dense,Add
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
import tensorflow as tf

from .units import residual_unit
from .units import attention_unit
from .units import resnet_layer

def AttentionResNet56(shape=(32,32,3), in_channel=32, kernel_size=5, n_classes=10, regularization=0.01):

    """
    This is to construct the Attention-56 Network
    
    Parameters
    ----------  
    shape:          The shape of input tensor for only one sample, and the default shape of all input tensor is: (batch size, 32, 32, 3).
    in_channel:     The number of filters for the Conv2D layer, this number is equal to the 'input_channel' paramter used in the following first residual_unit function. so we name it as in_channel here. 
    kernel_size:    The shape of the Conv2D kernel.
    n_classes:      The number of predicted classes.
    regularization: Weights for the regularization term.
    
    Returns:
    ----------  
    model: the constructed Attention-56 model
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
    x = attention_unit(x, skip=1) # (batch size, 4, 4, 512)

    
    x = residual_unit(x, input_channel=512, output_channel=1024)   # (batch size, 4, 4, 1024)
    x = residual_unit(x, input_channel=1024, output_channel=1024)  # (batch size, 4, 4, 1024)
    x = residual_unit(x, input_channel=1024, output_channel=1024)  # (batch size, 4, 4, 1024)

    x = AveragePooling2D(pool_size=4, strides=1)(x)  # (batch size, 1, 1, 1024)
    x = Flatten()(x) # (batch size, 1024)

    output = Dense(n_classes, kernel_regularizer=l2(regularization), activation='softmax')(x)

    model = Model(input_layer, output)

    return model

def AttentionResNet128(shape=(32,32,3), in_channel=32, kernel_size=5, n_classes=10, regularization=0.01):

    """
    This is to construct the Attention-128 Network
    
    Parameters
    ----------  
    shape:          The shape of input tensor for only one sample, and the default shape of all input tensor is: (batch size, 32, 32, 3).
    in_channel:     The number of filters for the Conv2D layer, this number is equal to the 'input_channel' paramter used in the following first residual_unit function. so we name it as in_channel here. 
    kernel_size:    The shape of the Conv2D kernel.
    n_classes:      The number of predicted classes.
    regularization: Weights for the regularization term.
    
    Returns:
    ----------  
    model: the constructed Attention-128 model
    """

    input_layer = Input(shape=(32,32,3))  # (batch size, 32, 32, 3)
    x = Conv2D(in_channel, kernel_size=kernel_size, padding='same')(input_layer)  # (batch size, 32, 32, 32)
    x = MaxPooling2D(pool_size=2)(x)  # (batch size, 16, 16, 32)

    # The 1st Attention Module
    x = residual_unit(x, input_channel=32, output_channel=128)  # (batch size, 16, 16, 128)
    x = attention_unit(x, skip=3) # (batch size, 16, 16, 128)
    x = attention_unit(x, skip=3) # (batch size, 16, 16, 128)

    # The 2nd Attention Module
    x = residual_unit(x, input_channel=128, output_channel=256, stride=2)  # (batch size, 8, 8, 256)
    x = attention_unit(x, skip=2) # (batch size, 8, 8, 256)
    x = attention_unit(x, skip=2) # (batch size, 8, 8, 256)

    # The 3rd Attention Module
    x = residual_unit(x, input_channel=256, output_channel=512, stride=2)  # (batch size, 4, 4, 512)
    x = attention_unit(x, skip=1) # (batch size, 4, 4, 512)
    x = attention_unit(x, skip=1) # (batch size, 4, 4, 512)

    
    x = residual_unit(x, input_channel=512, output_channel=1024)   # (batch size, 4, 4, 1024)
    x = residual_unit(x, input_channel=1024, output_channel=1024)  # (batch size, 4, 4, 1024)
    x = residual_unit(x, input_channel=1024, output_channel=1024)  # (batch size, 4, 4, 1024)

    x = AveragePooling2D(pool_size=4, strides=1)(x)  # (batch size, 1, 1, 1024)
    x = Flatten()(x) # (batch size, 1024)

    output = Dense(n_classes, kernel_regularizer=l2(regularization), activation='softmax')(x)

    model = Model(input_layer, output)

    return model


def ResNet(layers,shape=(32,32,3),n_classes=10):
    
    '''
    This is to construct ResNet-56
    
    Parameters
    ----------  
    shape:     shape of the input tensor
    layers:    number of the core convolutional layers
    n_classes: number of predicted classes
    
    Returns
    ----------
    model: the constructed ResNet-56 model
    
    reference: https://keras.io/zh/examples/cifar10_resnet/
    
    '''
    input_layer = Input(shape=(32,32,3)) 
    
    # define the model parameters
    n_filters = 16
    n_blocks  = int((layers - 2) / 9)
    
    # before splitting into 2 paths, resnet performs Conv2D with BN-ReLU on input 
    x = resnet_layer(inputs      = input_layer,
                     num_filters = n_filters,
                     conv_first  = True)
    
    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(n_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                n_filters_out = n_filters * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                n_filters_out = n_filters * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=n_filters,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=n_filters,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=n_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=n_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = tf.keras.layers.add([x, y])

        n_filters = n_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(n_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=input_layer, outputs=outputs)
    return model
    
    
    
    
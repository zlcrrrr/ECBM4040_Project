from tensorflow.keras.layers import Input,BatchNormalization,Conv2D,MaxPooling2D,Activation,AveragePooling2D,Flatten,Dropout,Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model

from .units import residual_unit
from .units import attention_unit, attention_unit_ca, attention_unit_sa

def AttentionResNet56(shape=(32,32,3), in_channel=32, kernel_size=5, n_classes=10, regularization=0.00, dropout=0.0):

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
    
    x = Activation('relu')(x)

    x = AveragePooling2D(pool_size=4, strides=1)(x)  # (batch size, 1, 1, 1024)
    x = Flatten()(x) # (batch size, 1024)

    x = Dropout(dropout)(x)
    output = Dense(n_classes, kernel_regularizer=l2(regularization), activation='softmax')(x)

    model = Model(input_layer, output)

    return model


def AttentionResNet56_ca(shape=(32,32,3), in_channel=32, kernel_size=5, n_classes=10, regularization=0.00, dropout=0.0):

    """
    This is to construct the Attention-56 Network using the channel attention type
    
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
    x = attention_unit_ca(x, skip=2) # (batch size, 16, 16, 128)

    # The 2nd Attention Module
    x = residual_unit(x, input_channel=128, output_channel=256, stride=2)  # (batch size, 8, 8, 256)
    x = attention_unit_ca(x, skip=1) # (batch size, 8, 8, 256)

    # The 3rd Attention Module
    x = residual_unit(x, input_channel=256, output_channel=512, stride=2)  # (batch size, 4, 4, 512)
    x = attention_unit_ca(x, skip=0) # (batch size, 4, 4, 512)
    
    x = residual_unit(x, input_channel=512, output_channel=1024)   # (batch size, 4, 4, 1024)
    x = residual_unit(x, input_channel=1024, output_channel=1024)  # (batch size, 4, 4, 1024)
    x = residual_unit(x, input_channel=1024, output_channel=1024)  # (batch size, 4, 4, 1024)
    
    x = Activation('relu')(x)

    x = AveragePooling2D(pool_size=4, strides=1)(x)  # (batch size, 1, 1, 1024)
    x = Flatten()(x) # (batch size, 1024)

    x = Dropout(dropout)(x)
    output = Dense(n_classes, kernel_regularizer=l2(regularization), activation='softmax')(x)

    model = Model(input_layer, output)

    return model


def AttentionResNet56_sa(shape=(32,32,3), in_channel=32, kernel_size=5, n_classes=10, regularization=0.00, dropout=0.0):

    """
    This is to construct the Attention-56 Network using the spatial attention type
    
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
    x = attention_unit_sa(x, skip=2) # (batch size, 16, 16, 128)

    # The 2nd Attention Module
    x = residual_unit(x, input_channel=128, output_channel=256, stride=2)  # (batch size, 8, 8, 256)
    x = attention_unit_sa(x, skip=1) # (batch size, 8, 8, 256)

    # The 3rd Attention Module
    x = residual_unit(x, input_channel=256, output_channel=512, stride=2)  # (batch size, 4, 4, 512)
    x = attention_unit_sa(x, skip=0) # (batch size, 4, 4, 512)
    
    x = residual_unit(x, input_channel=512, output_channel=1024)   # (batch size, 4, 4, 1024)
    x = residual_unit(x, input_channel=1024, output_channel=1024)  # (batch size, 4, 4, 1024)
    x = residual_unit(x, input_channel=1024, output_channel=1024)  # (batch size, 4, 4, 1024)
    
    x = Activation('relu')(x)

    x = AveragePooling2D(pool_size=4, strides=1)(x)  # (batch size, 1, 1, 1024)
    x = Flatten()(x) # (batch size, 1024)

    x = Dropout(dropout)(x)
    output = Dense(n_classes, kernel_regularizer=l2(regularization), activation='softmax')(x)

    model = Model(input_layer, output)

    return model

def AttentionResNet92(shape=(32,32,3), in_channel=32, kernel_size=5, n_classes=10, regularization=0.00, dropout=0.0):

    """
    This is to construct the Attention-92 Network
    
    Parameters
    ----------  
    shape:          The shape of input tensor for only one sample, and the default shape of all input tensor is: (batch size, 32, 32, 3).
    in_channel:     The number of filters for the Conv2D layer, this number is equal to the 'input_channel' paramter used in the following first residual_unit function. so we name it as in_channel here. 
    kernel_size:    The shape of the Conv2D kernel.
    n_classes:      The number of predicted classes.
    regularization: Weights for the regularization term.
    
    Returns:
    ----------  
    model: the constructed Attention-92 model
    """
    input_layer = Input(shape=(32,32,3))  # (batch size, 32, 32, 3)
    x = Conv2D(in_channel, kernel_size=kernel_size, padding='same')(input_layer)  # (batch size, 32, 32, 32)
    x = MaxPooling2D(pool_size=2)(x)  # (batch size, 16, 16, 32)

    # The 1st Attention Module
    x = residual_unit(x, input_channel=32, output_channel=128)  # (batch size, 16, 16, 128)
    x = attention_unit(x, skip=3) # (batch size, 16, 16, 128)

    # The 2nd Attention Module
    x = residual_unit(x, input_channel=128, output_channel=256, stride=2)  # (batch size, 8, 8, 256)
    x = attention_unit(x, skip=2) # (batch size, 8, 8, 256)
    x = attention_unit(x, skip=2) # (batch size, 8, 8, 256)

    # The 3rd Attention Module
    x = residual_unit(x, input_channel=256, output_channel=512, stride=2)  # (batch size, 4, 4, 512)
    x = attention_unit(x, skip=1) # (batch size, 4, 4, 512)
    x = attention_unit(x, skip=1) # (batch size, 4, 4, 512)
    x = attention_unit(x, skip=1) # (batch size, 4, 4, 512)

    
    x = residual_unit(x, input_channel=512, output_channel=1024)   # (batch size, 4, 4, 1024)
    x = residual_unit(x, input_channel=1024, output_channel=1024)  # (batch size, 4, 4, 1024)
    x = residual_unit(x, input_channel=1024, output_channel=1024)  # (batch size, 4, 4, 1024)
    
    x = Activation('relu')(x)

    x = AveragePooling2D(pool_size=4, strides=1)(x)  # (batch size, 1, 1, 1024)
    x = Flatten()(x) # (batch size, 1024)

    x = Dropout(dropout)(x)
    output = Dense(n_classes, kernel_regularizer=l2(regularization), activation='softmax')(x)

    model = Model(input_layer, output)

    return model




def AttentionResNet164(shape=(32,32,3), in_channel=32, kernel_size=5, n_classes=10, regularization=0.0, dropout=0.0):

    """
    This is to construct the Attention-164 Network
    
    Parameters
    ----------  
    shape:          The shape of input tensor for only one sample, and the default shape of all input tensor is: (batch size, 32, 32, 3).
    in_channel:     The number of filters for the Conv2D layer, this number is equal to the 'input_channel' paramter used in the following first residual_unit function. so we name it as in_channel here. 
    kernel_size:    The shape of the Conv2D kernel.
    n_classes:      The number of predicted classes.
    regularization: Weights for the regularization term.
    
    Returns:
    ----------  
    model: the constructed Attention-164 model
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

    # The 2nd Attention Module
    x = residual_unit(x, input_channel=128, output_channel=256, stride=2)  # (batch size, 8, 8, 256)
    x = attention_unit(x, skip=1) # (batch size, 8, 8, 256)
    x = attention_unit(x, skip=1) # (batch size, 8, 8, 256)
    x = attention_unit(x, skip=1) # (batch size, 8, 8, 256)
    x = attention_unit(x, skip=1) # (batch size, 16, 16, 128)

    # The 3rd Attention Module
    x = residual_unit(x, input_channel=256, output_channel=512, stride=2)  # (batch size, 4, 4, 512)
    x = attention_unit(x, skip=0) # (batch size, 4, 4, 512)
    x = attention_unit(x, skip=0) # (batch size, 4, 4, 512)
    x = attention_unit(x, skip=0) # (batch size, 4, 4, 512)
    x = attention_unit(x, skip=0) # (batch size, 4, 4, 512)

    
    x = residual_unit(x, input_channel=512, output_channel=1024)   # (batch size, 4, 4, 1024)
    x = residual_unit(x, input_channel=1024, output_channel=1024)  # (batch size, 4, 4, 1024)
    x = residual_unit(x, input_channel=1024, output_channel=1024)  # (batch size, 4, 4, 1024)
    
    x = Activation('relu')(x)

    x = AveragePooling2D(pool_size=4, strides=1)(x)  # (batch size, 1, 1, 1024)
    x = Flatten()(x) # (batch size, 1024)

    x = Dropout(dropout)(x)
    output = Dense(n_classes, kernel_regularizer=l2(regularization), activation='softmax')(x)

    model = Model(input_layer, output)

    return model

import tensorflow as tf
from tensorflow.keras.layers import Conv2D,UpSampling2D,MaxPooling2D,Add,Activation,Lambda,Multiply,BatchNormalization
from tensorflow.keras.regularizers import l2

def residual_unit(input_layer, input_channel=None, output_channel=None, stride=1):

    """
    This is to implement the pre-activation Residual Unit, the structure of which is shown as: 
    
    input >> Batch Normalization >> ReLU >> 1x1 Conv2D 
          >> Batch Normalization >> ReLU >> 3x3 Conv2D
          >> Batch Normalization >> ReLU >> 1x1 Conv2D + input (identify)
          >> output
      
    Parameters
    ----------  
    input_layer:    The input layer of pre-activation Residual Unit. It's a 4-D array with the shape of (batch_size, height, width, n_channel)
    input_channel:  Number of channels for input layer
    output_channel: Number of channels/feature maps for output layer. The 4-th dimension (channel number) of output matrix. 
    stride:         The strides of the convolution along the height and width, which is the # of pixels to move between 2 neighboring receptive fields.
    
    Returns
    ----------  
    output: The output layer of the pre-activation Residual Unit
    """
    
    # define number of channels
    if (input_channel is None)|(output_channel is None):
        output_channel = input_layer.shape[-1] # n_channel
        input_channel  = output_channel//4 # n_channel//4
    
    # The ResNet module
    x = BatchNormalization()(input_layer)
    x = Activation('relu')(x)
    x = Conv2D(input_channel, (1, 1))(x) # shape: (height,width,n_channel//4)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(input_channel, (3, 3), padding='same', strides=stride)(x) # shape: (height,width,n_channel//4)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(output_channel, (1, 1), padding='same')(x) # shape: (height,width,n_channel)

    # The identity module
    input_layer = Conv2D(output_channel, (1, 1), padding='same', strides=stride)(input_layer) # shape: (height,width,n_channel)
    output      = Add()([x, input_layer]) # shape: (height,width,n_channel)

    return output



def attention_unit(input_layer, skip):

    """
    This is to construct the Attention Module which consists of two branches, its structure is shown as:
    
    input >> pre-processing Residual Units (p times) >> split into two branches
            
          1. Trunk Branch
          >> Residual Units (t times) >> Trunk output: T(x)
            
          2. Soft Mask Branch
          >> the 1-st down-sampling >> Residual Units (r times) >> skip connections >> the 2-nd down-sampling >> Residual Units (2r times)
          >> the 1-st up-sampling   >> skip connections >> Residual Units (r times) >> the 2-nd up-sampling   >> Cov2D x2 >> sigmoid >> Mask output: M(x)
          
    >> combine branches: (1+M(x)) * T(x) >> Residual Units (p times) >> final output
          

    Parameters
    ----------  
    input_layer: The input layer of the Attention Module. It's a 4-D array with the shape of (batch_size, height, width, n_channel)
    skip:        The number of skip connections
    
    Returns
    ----------  
    output: Tht output of the Attention Module
    """

    # specify the parameters of attention module 
    p = 1 # the # of pre-processing Residual Units before branch splitting
    t = 2 # the # of pre-processing Residual Units within the trunk branch
    r = 1 # the # of pre-processing Residual Units within the mask branch
    
    # number of channels for input layer (=n_channel)
    input_channel = input_layer.shape[-1]
   

    # 1. pre-processing Residual Units before branch splitting
    for _ in range(p):
        input_layer = residual_unit(input_layer) # shape: (batch_size,height, width, n_channel)

    # 2. Trunk Branch
    out_trunk = input_layer
    for _ in range(t):
        out_trunk = residual_unit(out_trunk) # shape: (batch_size,height, width, n_channel)
   
    
    # 3. Soft Mask Branch
    ## The first down sampling
    out_mask = MaxPooling2D()(input_layer) # shape: (batch_size,height/2, width/2, n_channel)
    for _ in range(r):
        out_mask = residual_unit(out_mask) # shape: (batch_size,height/2, width/2, n_channel)
    
    ## Skip connections 
    skip_connections = []
    
    for i in range(skip-1):
        out_skipconnection = residual_unit(out_mask) 
        skip_connections.append(out_skipconnection) # shape: (batch_size,height/2, width/2, n_channel)

    ## The second down sampling
        out_mask = MaxPooling2D()(out_mask) # shape: (batch_size,height/4, width/4, n_channel)
        for _ in range(r):
            out_mask = residual_unit(out_mask) # shape: (batch_size,height/4, width/4, n_channel)
    
    ## reverse the sequence of skip connections
    skip_connections = list(reversed(skip_connections))

    for i in range(skip-1):
    ## The first upsampling
        for _ in range(r):
            out_mask = residual_unit(out_mask) # shape: (batch_size,height/4, width/4, n_channel)
        out_mask = UpSampling2D()(out_mask) # shape: (batch_size,height/2, width/2, n_channel)

    ## Skip connections
        out_mask = Add()([out_mask, skip_connections[i]]) # shape: (batch_size,height/2, width/2, n_channel)

    ## The second upsamplping
    for _ in range(r):
        out_mask = residual_unit(out_mask) # shape: (batch_size,height/2, width/2, n_channel)
    out_mask = UpSampling2D()(out_mask)    # shape: (batch_size,height, width, n_channel)

    ## Output of soft mask branch
    out_mask = Conv2D(input_channel, (1, 1))(out_mask) # shape: (batch_size,height, width, n_channel)
    out_mask = Conv2D(input_channel, (1, 1))(out_mask) # shape: (batch_size,height, width, n_channel)
    out_mask = Activation('sigmoid')(out_mask)         # shape: (batch_size,height, width, n_channel)

    # 4. Combine the trunk branch with soft mask branch: (1 + out_mask) * out_trunk
    out_mask = Lambda(lambda x: x + 1)(out_mask)
    output   = Multiply()([out_mask, out_trunk]) # shape: (batch_size,height, width, n_channel)

    # 5. Output of attention unit: add the last residual unit
    for _ in range(p):
        output = residual_unit(output) # shape: (batch_size,height, width, n_channel)

    return output


def resnet_layer(inputs,num_filters=16,kernel_size=3,strides=1,
                 activation='relu',batch_normalization=True,conv_first=True):
    """
    This is to construct the basic unit for ResNet-56, which is actually a 
    2D Convolution-Batch Normalization-Activation stack builder.

    Parameters
    ----------  
    inputs:      input variable from input image or previous layer (tensor)
    num_filters: the number of filters for the Conv2D layer (int)
    kernel_size: the size/dimension of the kernal for the Conv2D (int) 
    strides:     the strides of the convolution along the height and width, which is the # of pixels to move between 2 neighboring receptive fields.
    activation:  name of the activation layer (string)
    batch_normalization : whether to add a batch normalization layer (bool)
    conv_first : the sequence of Conv2D, BN, and activation layer (bool). 
                 If true, it would be Conv2D-BN-Activation, otherwise it would be BN-Activation-Conv2D
    Returns
    ----------  
    x: output of this resnet layer
        
    reference: https://keras.io/zh/examples/cifar10_resnet/
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x








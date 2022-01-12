from tensorflow.keras.layers import Input, Flatten, MaxPooling2D, Dense, Conv2D, Conv2DTranspose, Reshape, UpSampling2D, BatchNormalization

from tensorflow.keras.models import Model

def getModels(modelName, N=-1, activation='linear', activationDecoder='relu'):

    # if modelName=="72x72_Dense_512":
    #     inputs = Input(shape=(72*72),name='input')
    #     encoded = Dense(512, activation=activation)(inputs)
    #     decoded = Dense(72*72, activation=activation)(encoded)

    #     autoencoder = Model(inputs, decoded)

    # elif modelName=="72x72_Dense_256":
    #     inputs = Input(shape=(72*72),name='input')
    #     encoded = Dense(256, activation=activation)(inputs)
    #     decoded = Dense(72*72, activation=activation)(encoded)

    #     autoencoder = Model(inputs, decoded)

    # elif modelName=="72x72_Dense_128":
    #     inputs = Input(shape=(72*72),name='input')
    #     encoded = Dense(128, activation=activation)(inputs)
    #     decoded = Dense(72*72, activation=activation)(encoded)

    #     autoencoder = Model(inputs, decoded)

    # elif modelName=="72x72_Dense_64":
    #     inputs = Input(shape=(72*72),name='input')
    #     encoded = Dense(64, activation=activation)(inputs)
    #     decoded = Dense(72*72, activation=activation)(encoded)

    #     autoencoder = Model(inputs, decoded)

    if modelName=="72x72_Dense" and N>0:
        inputs = Input(shape=(72*72),name='input')
        encoded = Dense(N, activation=activation)(inputs)
        decoded = Dense(72*72, activation=activationDecoder)(encoded)

        autoencoder = Model(inputs, decoded)

    elif modelName=="72x72_Dense_BatchNorm" and N>0:
        inputs = Input(shape=(72*72),name='input')
        encoded = Dense(N, activation=activation)(inputs)
        encoded = BatchNormalization()(encoded)
        decoded = Dense(72*72, activation=activationDecoder)(encoded)

        autoencoder = Model(inputs, decoded)

    elif modelName=="72x72_Dense_1024_128":
        inputs = Input(shape=(72*72),name='input')
        hidden  = Dense(1024, activation=activation)(inputs)
        encoded = Dense(128, activation=activation)(hidden)
        hidden  = Dense(1024, activation=activation)(encoded)
        decoded = Dense(72*72, activation=activation)(hidden)

        autoencoder = Model(inputs, decoded)
    elif modelName=="72x72_Dense_128_128":
        inputs = Input(shape=(72*72),name='input')
        hidden  = Dense(1024, activation=activation)(inputs)
        encoded = Dense(128, activation=activation)(hidden)
        hidden  = Dense(1024, activation=activation)(encoded)
        decoded = Dense(72*72, activation=activation)(hidden)

        autoencoder = Model(inputs, decoded)
    else:
        print(f'Unknown model name {modelName}')
        exit()

    return autoencoder

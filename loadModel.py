from tensorflow.keras.models import model_from_json

def getModel(f_model, skipWeights=False):
    with open(f_model,'r') as f:
        if 'QActivation' in f.read():
            from qkeras import QDense, QConv2D, QActivation,quantized_bits,Clip
            f.seek(0)
            model = model_from_json(f.read(),
                                    custom_objects={'QActivation':QActivation,
                                                    'quantized_bits':quantized_bits,
                                                    'QConv2D':QConv2D,
                                                    'QDense':QDense,
                                                    'Clip':Clip})
        else:
            f.seek(0)
            model = model_from_json(f.read())

    if not skipWeights:
        hdf5  = f_model.replace('json','hdf5')
        model.load_weights(hdf5)

    return model

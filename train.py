import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from LossFunctions import r_weighted_mse, r2_weighted_mse, r3_weighted_mse, telescopeMSE, sqrt_mse, rMask_weighted_mse, rMask2_weighted_mse, rMask3_weighted_mse, mseTest, InvWeightedMSE

from models import getModels

from tensorflow.keras.callbacks import EarlyStopping

def trainModel(modelName, activation, activationDecoder, loss, data, N_epochs=100, N=-1, outputDir='outputs', outputName='model', noEarlyStop=False, batch_size=32, verbosity=0):
    if N>0:
        model = getModels(modelName, N=N, activation=activation, activationDecoder=activationDecoder)
        modelName = f'{modelName}_{N}'
    else:
        model = getModels(modelName, activation=activation)

    if loss=='r_weighted':
        lossFunction=r_weighted_mse
    elif loss=='r2_weighted':
        lossFunction=r2_weighted_mse
    elif loss=='r3_weighted':
        lossFunction=r3_weighted_mse
    elif loss=='rMask_weighted':
        lossFunction=rMask_weighted_mse
    elif loss=='rMask2_weighted':
        lossFunction=rMask2_weighted_mse
    elif loss=='rMask3_weighted':
        lossFunction=rMask3_weighted_mse
    elif loss=='mseTest':
        lossFunction=mseTest
    elif loss=='invweight':
        lossFunction=InvWeightedMSE
    elif loss.lower()=='telescope' or loss.lower()=="telescopemse":
        lossFunction=telescopeMSE
    elif loss.lower()=='sqrtmse' or loss.lower()=="sqrt_mse":
        lossFunction=sqrt_mse
    else:
        lossFunction=loss

    if noEarlyStop:
        callbacks=None
    else:
        callbacks = [EarlyStopping(monitor='loss', patience=5)]

    model.compile(optimizer='adam', loss=lossFunction)

    # Autoencoder
    model.summary()

    history = model.fit(data.reshape(-1,72**2),data.reshape(-1,72**2),
                        epochs=N_epochs,
                        verbose=verbosity,
                        callbacks=callbacks,
                        batch_size=batch_size,
                        validation_split=0.1)

    print(f'Trained for {len(history.history["loss"])} epochs')
    outputPath = f'{outputDir}/AE_{outputName}'
    print(outputPath)
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)

        
    fig,ax = plt.subplots(1,1,figsize=(10,5))
    ax.plot(history.history['loss'],label='Loss')
    ax.plot(history.history['val_loss'],label='Val Loss')
    ax.set_yscale('log')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    fig.legend()
    fig.savefig(f'{outputPath}/{outputName}_Loss.pdf')

    model.save_weights(f'{outputPath}/{outputName}.hdf5')
    with open(f'{outputPath}/{outputName}.json','w') as f:
        f.write(model.to_json())

    return model


def evalModel(model, data, outputDir='outputs', outputName='model', extraInfo='', dataPreProcessing='None'):

    output = model.predict(data.reshape(-1,72*72))

    if dataPreProcessing=='None':
        outputToSave =output.astype(complex)**0.5
    elif dataPreProcessing=='Norm':
        outputToSave =(output.astype(complex)**1024)**0.5
    else:
        outputToSave = output.astype(complex)

    outputPath = f'{outputDir}/AE_{outputName}'

    with h5py.File(f'{outputPath}/data{extraInfo}_cell_phase_AE_{outputName}.h5','w') as h5f:
        h5f.create_dataset('exchange/data',data=outputToSave.reshape(1,4488,72,72))

    N=[1873,34,676,1074,3557,2457,2320,3558]
    
    fig,ax = plt.subplots(len(N),3,figsize=(15,4*len(N)))
    for i in range(len(N)):
        im = ax[i][0].imshow(data[N[i]],norm=LogNorm())
        fig.colorbar(im,ax=ax[i][0])
        ax[i][0].set_title(f'Original (idx={N[i]})')
        
        im = ax[i][1].imshow(output[N[i]].reshape(72,72),norm=LogNorm())
        fig.colorbar(im,ax=ax[i][1])
        ax[i][1].set_title(f'AE Output (idx={N[i]})')

        im = ax[i][2].imshow(np.where(output[N[i]]<0.5, 0, output[N[i]]).reshape(72,72),norm=LogNorm())
        fig.colorbar(im,ax=ax[i][2])
        ax[i][2].set_title(f'AE Output ZS (idx={N[i]})')
    
    fig.savefig(f'{outputPath}/diffraction_comparison_{outputName}{extraInfo}.pdf')
    



if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--loss', dest='loss', default='mse',
                        help='loss function to use in training')
    parser.add_argument('--act1', dest='activationEncoder', default='linear', choices=['linear','relu','sigmoid'],
                        help='final activation function to use on the decoded values')
    parser.add_argument('--act2', dest='activationDecoder', default='linear', choices=['linear','relu'],
                        help='final activation function to use on the decoded values')
    parser.add_argument('--model', dest='modelName', default='72x72_Dense_512',
                        help='name of defined model to use')
    parser.add_argument('--Nepochs', dest='N_epochs', default=200, type=int,
                        help='number of epochs')
    parser.add_argument('--verbosity', dest='verbosity', default=0, type=int,
                        help='Verbosity of training output')
    parser.add_argument('--batch_size', dest='batch_size', default=32, type=int,
                        help='Batch size to use in training')
    parser.add_argument('--dataPre', dest='dataPreProcessing', default='None',
                        help='Preprocessing to perform on data: options are "None": use photon counts (default), "Sqrt": use sqrt of photon counts, "Log": take log2 of photon counts')
    parser.add_argument('-N', dest='N_latent', action='append', type=int,
                        help='Number of nodes in latent space, appendable list to train multiple models at once')
    parser.add_argument('--noEarlyStop', dest='noEarlyStop', action='store_true', default=False,
                        help='Skip early stopping')
    args = parser.parse_args()

    with h5py.File(f"TrainingData/data_cell_phase_n2e7.h5", "r") as f:
        data = f['exchange/data'][...][0].real

    with h5py.File(f"TrainingData/data_cell_phase_n2e7_ref.h5", "r") as f:
        dataRef = f['exchange/data'][...][0].real


    activationDecoder=args.activationDecoder

    outputDir = ''
    if args.dataPreProcessing=='None':
        data = data.real**2
        dataRef = dataRef.real**2
        outputDir = f'outputs_PhoCount_{activationDecoder}'
    elif args.dataPreProcessing=='Sqrt':
        outputDir = f'outputs_Sqrt_{activationDecoder}'
    elif args.dataPreProcessing=='Norm':
        outputDir = f'outputs_Norm_{activationDecoder}'
        data = (data.real**2)/1024.
        dataReg = (dataRef.real**2)/1024.
#        outputDir = 'outputs'
    else:
        print('Unknown preprocessing step')
        exit()

    if not args.N_latent is None:
        N_latent = args.N_latent
    else:
        N_latent = [1,10,30,50,100,300,500,1000,1500,2000,2500,3000,3500,4000]


    print(N_latent)
    for modelName in ['72x72_Dense']:
#        for N in [4000]:
        for N in N_latent:
            for loss in [args.loss]:
                for activation in [args.activationEncoder]:
                    outputName=f'{modelName}_{N}_{activation}_{loss.lower()}'

                    model = trainModel(modelName=modelName, 
                                       activation=activation, 
                                       activationDecoder=activationDecoder, 
                                       loss=loss,
                                       data=data,
                                       N_epochs=args.N_epochs,
                                       verbosity=args.verbosity,
                                       batch_size=args.batch_size,
                                       N=N, 
                                       noEarlyStop=args.noEarlyStop,     
                                       outputDir=outputDir,
                                       outputName=outputName)

                    evalModel(model=model, 
                              data=data, 
                              outputDir=outputDir,
                              outputName=outputName,
                              dataPreProcessing=args.dataPreProcessing)

                    evalModel(model=model,
                              data=dataRef,
                              outputDir=outputDir,
                              outputName=outputName,
                              extraInfo='_ref',
                              dataPreProcessing=args.dataPreProcessing)

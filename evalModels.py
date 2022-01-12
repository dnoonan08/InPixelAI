from loadModel import getModel
import h5py

with h5py.File(f"TrainingData/data_cell_phase_n2e7.h5", "r") as f:
    evalData = f['exchange/data'][...][0].real**2
    
with h5py.File(f"TrainingData/data_cell_phase_n2e7_ref.h5", "r") as f:
    evalDataRef = f['exchange/data'][...][0].real**2
    
N=[1,10,30,50,100,300,500,1000,1500,2000,2500]

for n in N:
    print(n)
    fName=f'outputs_PhoCount/AE_72x72_Dense_{n}_linear_mse/72x72_Dense_{n}_linear_mse.json'
    model=getModel(fName)

    output = model.predict(evalData.reshape(-1,72*72))**0.5
    outputRef = model.predict(evalDataRef.reshape(-1,72*72))**0.5

    outName=f'outputs_PhoCount/AE_72x72_Dense_{n}_linear_mse/data_cell_phase_AE_72x72_Dense_{n}_linear_mse.h5'
    with h5py.File(outName,'w') as h5f:
        h5f.create_dataset('exchange/data',data=output.reshape(1,4488,72,72))

    outName=f'outputs_PhoCount/AE_72x72_Dense_{n}_linear_mse/data_ref_cell_phase_AE_72x72_Dense_{n}_linear_mse.h5'
    with h5py.File(outName,'w') as h5f:
        h5f.create_dataset('exchange/data',data=outputRef.reshape(1,4488,72,72))

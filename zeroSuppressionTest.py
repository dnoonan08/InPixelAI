import os
import numpy as np
import h5py


loss = ['mse','msle']
activation=['linear','relu']
nodes=[50,100]


for N in nodes:
   for a in activation:
      for l in loss:
         fName = f'outputs_PhoCount_{a}/AE_72x72_Dense_{N}_linear_{l}/data_cell_phase_AE_72x72_Dense_{N}_linear_{l}.h5'
         with h5py.File(fName, "r") as f:
            data = f['exchange/data'][...]

         for x in ['0000','0001','0010','0100','0250','0500','1000']:
            y = int(x)/1000.
            print(y,x, fName)

            outputDir = f'outputs_ZeroSuppressed/outputs_PhoCount_{a}/AE_72x72_Dense_{N}_linear_{l}_ZS_{x}'
            os.mkdir(outputDir)

            with h5py.File(f'{outputDir}/data_cell_phase_AE_72x72_Dense_{N}_linear_{l}_ZS_{x}.h5','w') as h5f:
               dataZS = np.where(data<y,0,data)
               h5f.create_dataset('exchange/data',data=dataZS.reshape(1,4488,72,72))

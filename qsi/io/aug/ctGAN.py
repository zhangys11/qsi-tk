import random 
import pandas as pd
import numpy as np
from ctgan import CTGAN # CTGANSynthesizer
import matplotlib.pyplot as plt

def expand_dataset(X, y, NX = 3, epochs = 10, batch_size = 100, cuda = True, verbose = True):
    '''
    use ctGAN to upsample

    batch_size : default to 500. MUST be N * pac. pac (int) is the number of samples to group together when applying the discriminator. (Defaults to 10)
    ''' 
    Xnew = np.array(X).copy()
    ynew = list(y)

    for c in set(y):     

        Xc = X[y == c]
       
        ctgan = CTGAN(generator_dim=(128,128), discriminator_dim=(128,128), epochs = epochs, batch_size = batch_size, cuda = cuda)
        ctgan.fit(Xc, discrete_columns = [i for i in range(Xc.shape[1])])

        if verbose:
            print('--- ctgan trained for y = ' + str(c) + '---')
            print(ctgan.__dict__)

        # Synthetic copy
        Xce = ctgan.sample(len(y[y == c]) * NX)

        if False:
            print('--- Below are generated samples for y = ' + str(c) + ' ---')
            for x in Xce.values:
                plt.plot(x)
                plt.axis('off')
                plt.show()

        Xnew = np.vstack((Xnew, Xce))
        ynew = ynew + [c]*len(y[y == c]) * NX

    return Xnew,ynew
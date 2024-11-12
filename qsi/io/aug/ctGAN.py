import random 
import pandas as pd
import numpy as np
from ctgan import CTGAN # CTGANSynthesizer
import matplotlib.pyplot as plt

def expand_dataset(X, y, nobs, 
                   embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256), 
                   epochs=20, batch_size=100, cuda = True, verbose=False):
    '''
    use ctGAN to upsample

    nobs : how many samples to generate.
    batch_size : default to 500. MUST be N * pac. pac (int) is the number of samples to group together when applying the discriminator. (Defaults to 10)
    '''
    
    train_data = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1)
    conditions = [train_data.columns[-1]]
    
    ctgan = CTGAN(batch_size=batch_size,
                  embedding_dim=embedding_dim, generator_dim=generator_dim, discriminator_dim=discriminator_dim,
                  cuda=cuda, verbose = True)
    ctgan.fit(train_data,
              discrete_columns = conditions, # [train_data.shape[1]-1] # Default use y. discrete_columns (list-like): List of discrete columns to be used to generate the Conditional Vector. If train_data is a Numpy array, this list should contain the integer indices of the columns. Otherwise, if it is a pandas.DataFrame, this list should contain the column names.
              epochs=epochs)
    
    if verbose:

        import torch
        from torchviz import make_dot
        import IPython.display

        # model_path = base64.b64encode(os.urandom(32))[:8].decode("utf-8")+'.pth'
        # ctgan.save(model_path)
        # tmp_model = torch.load(model_path)
        
        input_vec = torch.zeros(batch_size, ctgan._embedding_dim + len(conditions), dtype=torch.float, requires_grad=False)
        IPython.display.display(make_dot(ctgan._generator(input_vec.to('cuda') if cuda else input_vec)))
    
    synthetic_data = ctgan.sample(nobs)
    X_new = synthetic_data.iloc[:, :-1]
    y_new = synthetic_data.iloc[:, -1]
    return X_new, y_new


def expand_dataset_stratified(X, y, NX = 3, epochs = 10, batch_size = 100, cuda = True, verbose = True):
    '''
    A stratified version of ctgan aug. It uses all features as condition vector?
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
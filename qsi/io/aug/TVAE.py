'''
Tabular VAE
'''

import os
import pandas as pd
from ctgan import TVAE

def expand_dataset(X,y,nobs,
                   embedding_dim=128,compress_dims=(128, 128),decompress_dims=(128, 128),
                   epochs = 20, cuda=True, verbose=False):
    train_data=pd.concat([X, y], axis=1)
    tvae = TVAE(epochs=epochs, 
                embedding_dim=embedding_dim,compress_dims=compress_dims,decompress_dims=decompress_dims,
                cuda=cuda, verbose = verbose)
    
    tvae.fit(train_data)

    if verbose: # show model structure

        # from keras.utils import plot_model
        import torch
        from torchviz import make_dot
        import IPython.display
        # import base64

        input_vec = torch.zeros(1, tvae.embedding_dim, dtype=torch.float, requires_grad=False).to('cuda')
        IPython.display.display('<b>TVAE decoder</b>')
        IPython.display.display(make_dot(tvae.decoder(input_vec)))

        # model_path = base64.b64encode(os.urandom(32))[:8].decode("utf-8")+'.pth'
        # tvae.decoder.save(model_path)
        # tmp_model = torch.load(model_path)
        # input_vec = torch.zeros(1, X.shape[1], dtype=torch.float, requires_grad=False) #.to('cuda')
        # IPython.display.display(make_dot(tmp_model(input_vec.to('cuda') if cuda else input_vec)))      

    synthetic_data = tvae.sample(nobs)
    X_new = synthetic_data.iloc[:, :-1]
    y_new = synthetic_data.iloc[:, -1]

    return X_new, y_new
import numpy as np
from sklearn.preprocessing import StandardScaler

def expand_dataset(X, y, NX = 3):
    '''
    Parameters
    ----------
    NX : how many times to expand the dataset.
    '''

    # use SMOTE to upsample
    labels = set(y)

    for label in labels:
        X_grp = X[y == label]        
        new_data = create_random_samples(X_grp, sample_size=len(y[y == label]) * NX)            

        X = np.vstack((X, new_data))
        y = np.append(y, [label] * len(y[y == label]) * NX)

    return X, y

def create_random_samples(X, sample_size=1, batch_size = 32, h_dim=200, z_dim = 10, save_path=None):
    '''
    Train a VAE model with one hidden layer and generate random samples from its decoder.

    Parameters
    ----------
    X : dataset.
    sample_size : how many samples to generate.
    batch_size : batch size for training.
    h_dim : hidden layer dimension.
    z_dim : latent dimension.
    save_path : path to save the trained model.
    '''

    from cs1.basis.adaptive.vae import train_vae # cs1 version should >= 0.2.2
    import torch
    
    scaler = StandardScaler()  # MinMaxScaler() # StandardScaler()
    X = scaler.fit_transform(X)

    n = X.shape[1]
    h_dim1 = h_dim
    h_dim2 = 0

    model = train_vae(X, batch_size=batch_size,
                      h_dim1=h_dim1, h_dim2=h_dim2, z_dim=z_dim)
    
    if save_path is not None and save_path != '':
        if not save_path.endswith('.pth'):
            save_path = save_path + '.pth'
        torch.save(model.state_dict(), save_path)

    ### show model structure
    # input_vec = torch.zeros(1, n, dtype=torch.float, requires_grad=False).to('cuda')
    # out = model1(input_vec)
    # display(make_dot(out))

    with torch.no_grad():
        # Generating 64 random z in the representation space
        z = torch.randn(sample_size, z_dim).cuda()
        # Evaluating the decoder on each of them
        sample = model.decoder(z).cuda().cpu().numpy() # Use Tensor.cpu() to copy the tensor to host memory first.

    return scaler.inverse_transform( sample )
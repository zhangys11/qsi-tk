import numpy as np
import matplotlib.pyplot as plt

def raman_prior():
    '''
    Return Raman prior knowledge, i.e., what wavenumber ranges correspond to what functional groups (chemical bonds).
    '''

    d = {}

    d['alkane_1'] = list(range(1295, 1305 + 1))
    d['alkane_2'] = list(range(800, 900 + 1)) + list(range(1040, 1100 + 1))
    d['branched_alkane_1'] = list(range(900, 950 + 1))
    d['branched_alkane_2'] = list(range(1040, 1060 + 1))
    d['branched_alkane_3'] = list(range(1140, 1170 + 1))
    d['branched_alkane_4'] = list(range(1165, 1175 + 1))
    d['haloalkane_1'] =  list(range(605, 615 + 1))
    d['haloalkane_2'] =  list(range(630, 635 + 1))
    d['haloalkane_3'] =  list(range(655, 675 + 1))
    d['haloalkane_4'] =  list(range(740, 760 + 1))
    d['alkene'] = list(range(1638, 1650 + 1))
    d['alkyne'] = list(range(2230, 2237 + 1))
    d['toluence'] = list(range(990, 1010 + 1))
    d['alcohol'] = list(range(800, 900 + 1))
    d['aldehyde'] = list(range(1725, 1740 + 1))
    d['ketone'] = list(range(1712, 1720 + 1))
    d['ether'] = list(range(820, 890 + 1))
    d['carboxylic_acid'] = list(range(820, 890 + 1))
    d['ester'] = list(range(634, 644 + 1))
    d['amine_1'] = list(range(740, 833 + 1))
    d['amine_2'] = list(range(1000, 1250 + 1))
    d['amide'] = list(range(700, 750 + 1))
    d['nitrile'] = list(range(2230, 2250 + 1))

    return d

def plot_raman_prior():

    d = raman_prior()

    plt.figure(figsize = (14,7))

    for idx, key in enumerate(d):
        # print(d[key])
        plt.scatter(d[key], [-idx] * len(d[key]), lw = 5, label = key)
        
    plt.legend(loc = "upper right")
    plt.yticks([])
    plt.xticks(range(500, 3001, 500))
    plt.show()

def binning_op(xaxis, region, filter = 'rbf', SD = 1):
    '''
    xaxis : the entire x axis range, e.g., [0, 3000]
    region : e.g., [100,200]
    filter : can be 'rbf', 'sinc', 'logistic', 'uniform'. Uniform is just averaging filter.
    SD : for rbf kernel, the region will lie inside +/-SD
    
    Return : op array. Has the length of xaxis.
    '''
    if filter == 'uniform':
        op = np.ones(len(xaxis)) / len(xaxis)
    # todo: others
    return op


def adaptive_binning(X, regions, filter = 'rbf'):
    '''
    Convert one data to binned features.
    Break down the axis as sections. Each seection is an integral of the signal intensities in the region.
    Integration can be done by radius basis function / sinc kernel, etc.

    filter : Apply a filter operator to a continuous region. Can be 'rbf', 'sinc', 'logistic', 'uniform'. Uniform is just averaging filter.
    '''

    Fss = []
    for x in X:

        Fs = [] # the discrete features for one data sample
        for region in regions:
            op = binning_op([0, len(x)], region, filter)
            F = (op*x).sum()
            Fs.append(F)

        Fss.append(Fs)

    return np.array(Fss)


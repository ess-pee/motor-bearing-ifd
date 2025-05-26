import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from utility_functions import snipper, extractor, asinput
from data_extraction_functions import extract_cwru_data, extract_mfd_data, extract_tri_data

def preprocess(dataset: str = 'cwru'):

    # I don't know why anyone would want to use this script, the files needed for this cannot be uploaded to the internet but I've just done this for amusement.
    # I mean I wouldn't touch this script with a 10 foot pole, but if you truly want to parlay, I've implemented
    # the handling of the dirty work already for you

    # Also if you somehow get hold of the way I've sorted the files, ALL 2.5 GIGS of them, then you can probably tweak this script to work just like you want it.
    # Otherwise you'll just need to trust me that indeed Shaurya Pathak does know what he's doing.

    if dataset == 'cwru':
        data, classes, sample_rate = extract_cwru_data()
    elif dataset == 'mfd':
        data, classes, sample_rate = extract_mfd_data()
    elif dataset == 'tri':    
        data, classes, sample_rate = extract_tri_data()
    else:
        return 'Invalid dataset entered. Please choose either "cwru", "mfd", or "tri".'

    # Some necessary variables and ancillaries
    minmax = MinMaxScaler(feature_range=(-1,1))
    window_size = int(sample_rate/10)
    stride = int(window_size/2)

    data = {key : [snipper(value) for value in values] for key, values in data.items() for value in values}

    data = {key : [minmax.fit_transform(np.reshape(value, (-1,1))) for value in values] for key, values in data.items()} 

    data = {key : [[value[i:i + window_size] for i in range(0, len(value) - window_size + 1, stride)] for value in values] for key, values in data.items()} 

    data = {key : np.squeeze([leaf for tree in data[key] for leaf in tree]) for key in data}

    dfs = []
    for key, values in data.items():
        tdf = pd.DataFrame(values)
        tdf['label'] = key
        dfs.append(tdf)

    df = pd.concat(dfs, ignore_index=True)

    traindf, testdf = train_test_split(df, test_size=0.2, random_state=0, shuffle=True)

    xtrain, ytrain = extractor(traindf)
    xtest, ytest = extractor(testdf)

    ytest = pd.get_dummies(ytest)
    ytrain = pd.get_dummies(ytrain)

    enc_ord = ytrain.columns.tolist()

    xtrain = asinput(xtrain)
    ytrain = np.squeeze(asinput(ytrain.astype(int)))

    xtest = asinput(xtest)
    ytest = np.squeeze(asinput(ytest.astype(int)))

    return xtrain, ytrain, xtest, ytest, classes, window_size, enc_ord

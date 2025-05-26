import numpy as np

from data_preprocessing import preprocess

if __name__ == "__main__":
    np.random.seed(69) # yes I am a child at heart

    DATASET = 'cwru' # pick any from 'cwru', 'tri' or 'mfd'

    xtrain, ytrain, xtest, ytest, classes, window_size, enc_ord = preprocess(DATASET)

    randices = np.random.randint(0, len(xtest), size=10)
    xtest_samples = xtest[randices]
    ytest_samples = ytest[randices]

    np.savez(f'models/{DATASET}_samples.npz', smpl_num=randices, xsmpl=xtest_samples, ysmpl=ytest_samples) # I am aware of the atrocious nomenclature, I am not sorry
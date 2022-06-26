import numpy as np
from adaptation import AdaptiveBinning

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis = 1, keepdims = True)


if __name__ == "__main__":
    import numpy as np
    data = np.load('/home/tuananh/tuananh/AdaptiveBinning/raw_output.npy')

    probability = softmax(data[:, 1:])
    prediction = np.argmax(probability, axis=1)
    label = data[:, 0]

    infer_results = []

    for i in range(len(data)):
        correctness = (label[i] == prediction[i])
        infer_results.append([probability[i][prediction[i]], correctness])

    # Call AdaptiveBinning.
    AECE, AMCE, confidence, accuracy, cof_min, cof_max = AdaptiveBinning(infer_results, True)

    print('ECE based on adaptive binning: {}'.format(AECE))
    print('MCE based on adaptive binning: {}'.format(AMCE))

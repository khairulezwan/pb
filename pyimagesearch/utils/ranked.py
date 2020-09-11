# import packages

def rank5_accuracy(preds, labels):
    # init rank 1 and rank 5 acc
    rank1 = 0
    rank5 = 0

    # loop over the predictions and g-turth labels
    for(p, gt) in zip(preds, labels):
        # sort the probabilitie by their index in descending
        # order so that the more confident guesses are at
        # the front of the list
        p = np.sort(p)[::-1]

        # check if the g-truth label is top 5
        if gt in p[:5]:
            rank5 += 1

        # check to see if the g-truth is the 1 prediction
        if gt == p[0]:
            rank1 += 1

    # compute the final rank-1 and rank-5 accuracies
    rank1 /= float(len(preds))
    rank5 /= float(len(preds))

    # return a tuple of the rank-1 and rank-5 accuracies
    return(rank1, rank5)
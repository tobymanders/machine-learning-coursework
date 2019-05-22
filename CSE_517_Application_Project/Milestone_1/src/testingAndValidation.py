def runCrossValidation(untrainedModel,folds):
    X,y = loadTrainData()
    scores = np.sqrt(-cross_val_score(untrainedModel, X, y, cv=folds,scoring="neg_mean_squared_error"))
    print("Error: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return scores

def runExperiment(model, test_size=.2):
    '''
    Runs a model on the training data by splitting it into train/validation sets
    input:
        model: model to be run - function taking in (Xtr,yTr,Xte) returning predictions
    '''
    X,y = loadTrainData()
    xTr,xVal, yTr, yVal = train_test_split(X,y,test_size = test_size)
    preds = model(xTr,yTr,xVal)
    error = rmse(preds,yVal)
    print("Error of {} \n".format(error))
    return error

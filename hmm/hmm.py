from hmmlearn.hmm import MultinomialHMM

def buildHMM(HMMFactory):

    model = MultinomialHMM(n_components=2, n_iter=200)
    model.startprob_ = HMMFactory.hiddenProb()
    model.transmat_ = HMMFactory.transMatrix()
    model.emissionprob_ = HMMFactory.emissionMatrix()
    return model



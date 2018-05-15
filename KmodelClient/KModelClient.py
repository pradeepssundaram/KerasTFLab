
class KModelClient(object):
    def __init__(self,X,Y,Tag,LossFunction):
        self._X = X
        self._Y = Y
        self._Tag = Tag
        self._LossFunction= LossFunction

    def RunExperiment(self):
        #TODO : Call ModelSelector.RunExperiment(X,Y,Tag)
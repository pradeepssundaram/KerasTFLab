import CoreNetwork.CoreNetworkModel as cn
import CoreNetwork.ModelParameters as mp
import CoreNetwork.Optimizers as asop
import tensorflow as tf
import json
import numpy as np


class KModelSelector(object):

    def __init__(self):
        self._configs = self._getconfigs()
        self._learning_rate = 0.01
        self._optimizers = self._getOptimizers()
        self._epochs = 5
        self._minibatchsize = 200


    def _getOptimizers(self):
        opts = [
            tf.train.GradientDescentOptimizer(learning_rate=self._learning_rate),
            asop.ASGradientDescentOptimizer(base_learning_rate=self._learning_rate),
            tf.train.RMSPropOptimizer(learning_rate=self._learning_rate),
            asop.ASRMSPropOptimizer(base_learning_rate=self._learning_rate),
            tf.train.AdamOptimizer(learning_rate=self._learning_rate),
            tf.train.MomentumOptimizer(learning_rate=self._learning_rate, momentum=.9),
            tf.train.MomentumOptimizer(learning_rate=self._learning_rate, momentum=.9, use_nesterov=True),
            tf.train.AdagradOptimizer(learning_rate=self._learning_rate)
        ]
        opt_names = ['SGD',
                     'SGD+AS',
                     'RMSProp',
                     'RMSProp+AS',
                     'ADAM',
                     'SGD+M',
                     'SGD+NM',
                     'Adagrad'
                     ]
        return zip(opts,opt_names)

    def _getconfigs(self):
        fpath = "D:\\BecomingADS\\KerasTFLab\\KModelSelector\\ModelSelectionConfig.json"
        with open(fpath) as f:
            _configs = json.load(f)
            return _configs

    def GetExperiments(self,Tag):
        tagconfigs = self._configs
        tagfns = tagconfigs[Tag]
        return tagfns

    def RunExperiments(self,Tag, X, Y, features, num_classes):
        tagf = self.GetExperiments(Tag)
        for l in tagf:
            getattr(KModelSelector,l)(self, X, Y, features, num_classes)

    def _train(self, modelparameters, X, Y):
        losses = []
        for optimizer, optimizername in (self._optimizers):
            init1 = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init1)
                network = cn.CoreNetwork(modelparameters)
                loss = network.train(sess, X, Y, self._epochs, optimizer, self._minibatchsize, optimizername)
                losses.append(loss)
            tf.reset_default_graph()

        return losses

    def BuildLogisticRegressionModel(self,X,Y,features,num_classes):
        model_p = mp.ModelParameters(input_feature_count=features, layers=[], activations=[], outputclasses=num_classes)
        losses = self._train(model_p,X,Y)
        return losses


    def BuildANNModel2Layers(self,X,Y,features,num_classes):
        model_p = mp.ModelParameters(input_feature_count=features, layers=[20, 5], activations=["relu", "relu"],
                                     outputclasses=num_classes)
        losses = self._train(model_p, X, Y)
        return losses



    def BuildANNModel3Layers(self,X,Y,features,num_classes):
        model_p = mp.ModelParameters(input_feature_count=features, layers=[20, 5, 10], activations=["relu", "relu","relu"],
                                     outputclasses=num_classes)
        losses = self._train(model_p, X, Y)
        return losses

    def BuildConvNet1(self,X,Y,features,num_classes):
        return "return Convnet 1"

    def BuildConvNet2(self,X,Y,features,num_classes):
        return "return Convnet 2"


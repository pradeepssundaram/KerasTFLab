
class ModelParameters(object):

    def __init__(self, input_feature_count, layers, activations, outputclasses):
        assert len(layers) == len(activations), "Number of layers should match number of activations"
        self.inputfeaturecount = input_feature_count
        #TO DO ASSERT if len (layers) <> len(activation)
        self.layers = layers
        self.activations = activations
        self.outputclasses = outputclasses

# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 20:49:32 2018

@author: Arthur P. Guillaumin
"""
import tensorflow as tf
from matplotlib import pyplot as plt

class CNNanalysis(object):
    """A class to analyse a trained CNN by plotting the kernels of 
    the different convolutional layers as well as the activations of each 
    layer.
    For this class to be used, the predictions of the customized estimator must
    return all layers, with names that will be used in this class for
    selection.
    """
    def __init__(self, estimator):
        """
        Args:
            estimator:          tf.ops.Estimator
                                A custom estimator
        """
        self.estimator = estimator
    
    def getLayerFilters(self, layerName):
        """This method retunrs the filters corresponding to a given layer.
        Args:
            layerName           str
                                The name used in the customized model_fun which
                                had been passed to the estimator."""
        var_names = self.estimator.get_variable_names()
        full_name = layerName + '/kernel'
        if full_name not in var_names:
            error = 'Layer {} not found in the model.'.format(layerName)
            raise(NameError(error))
        return self.estimator.get_variable_value(full_name)
    
    def plotLayerFilters(self, layerName, filters = None):
        """This method plots the filters corresponding to a given layer.
        Args:
            layerName           str
                                The name used in the customized model_fun which
                                had been passed to the estimator.
            filters             int list
                                List of filters to plot. By default, only the
                                first filter is plotted"""
        if filters is None:
            filters = [0]
        for i in filters:
            plt.figure()
            plt.title('Filter nb {}'.format(i))
            plt.imshow(self.getLayerFilters(layerName)[:,:,0,i].reshape((5,5)))
    
    def plotActivations(self, layerName, input_fn, filters = None):
        """Plots the activations for a given layer for different inputs.
        Args:
            layerName           str
                                The name of the layer. The corresponding value
                                must be passed through the EstimatorSpec when
                                mode is set to GraphKeys.PREDICT.
            input_fn            function ref
                                The input_fn used for inputs.
            filters             int list
                                A list of filter channels. By default, 0, i.e.
                                only the first channel is plotted.
            """
        #TODO: Find a better interface for this
        if filters is None:
            filters = [0]
        plt.figure()
        predictions_generator = self.estimator.predict(input_fn)
        while True:
            for i,pred in enumerate(predictions_generator):
                inpt = pred['input']
                plt.subplot(1+len(filters), 5, i+1)
                plt.imshow(inpt.reshape((28, 28)))
                for j,f in enumerate(filters):
                    plt.subplot(1+len(filters), 5, (j+1)*5+i+1)
                    plt.imshow(pred[layerName][:,:, f].reshape((28,28)))
                plt.pause(0.05)
                if i>=4:
                    break
            if input('Enter anything to quit') is not '':
                break
    
            
            
        
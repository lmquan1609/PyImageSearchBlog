from keras.callbacks import LambdaCallback
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import tempfile

class LearningRateFinder:
    def __init__(self, model, stop_factor=4, beta=0.98):
        # store the model, stop factor, and beta value (for computing a smoothed, average loss)
        self.model = model
        self.stop_factor = stop_factor
        self.beta = beta

        # initialize our list of lr and losses respectively
        self.lrs = []
        self.losses = []

        # initialize lr multiplier, average loss, best loss found thus far, current batch number, and weight file
        self.lr_mult = 1
        self.avg_loss = 0
        self.best_loss = 1e9
        self.batch_num = 0
        self.weights_file = None
    
    def reset(self):
        # re-initialize all variables from our constructor
        self.lrs = []
        self.losses = []
        self.lr_mult = 1
        self.avg_loss = 0
        self.best_loss = 1e9
        self.batch_num = 0
        self.weights_file = None

    def is_data_iter(self, data):
        # define the set of class types we will check for
        iter_classes = ['NumpyArrayIterator', 'DirectoryIterator', 'DataFrameIterator', 'Iterator', 'Sequence']

        return data.__class__.__name__ in iter_classes

    def on_batch_end(self, batch, logs):
        # grab the current learning rate and add log it to the lr that we've tried
        lr = K.get_value(self.model.optimizer.lr)
        self.lrs.append(lr)

        # grab the loss at the end this batch, increase total number of batches processed, compute the average loss, smooth it, and update the losses list with the smoothed value
        l = logs['loss']
        self.batch_num += 1
        self.avg_loss = (self.beta * self.avg_loss) + ((1 - self.beta) * l)
        smooth = self.avg_loss / (1 - (self.beta ** self.batch_num))
        self.losses.append(smooth)

        # compute the maximum loss stopping factor value
        stop_loss = self.stop_factor * self.best_loss

        # check to see whether the loss has grown too large
        if self.batch_num > 1 and smooth > stop_loss:
            # stop returning and return from the method
            self.model.stop_training = True
            return

        # check to see if the best loss should be updated
        if self.batch_num == 1 or smooth < self.best_loss:
            self.best_loss = smooth
        
        # increase the lr
        lr *= self.lr_mult
        K.set_value(self.model.optimizer.lr, lr)

    def find(self, train_X, start_lr, end_lr, epochs=None, steps_per_epoch=None, batch_size=32, sample_size=2048, verbose=2):
        # reset class-specific variables
        self.reset()

        # determine if we are using a data generator or not
        use_gen = self.is_data_iter(train_X)

        # if we're using a generator and steps per epoch is not supplied, raise an error
        if use_gen and steps_per_epoch is None:
            msg = 'Using generator without supplying steps_per_epoch'
            raise Exception(msg)

        # if we're not using a generator then our entire dataset must already be in memory
        elif not use_gen:
            # grab the number of samples in the training data and then derive the number of steps per epoch
            num_samples = len(train_X[0])
            steps_per_epoch = num_samples / float(batch_size)

        # if no number of training epochs are supplied, compute the training epochs based on a default sample size
        if epochs is None:
            epochs = int(np.ceil(sample_size / float(batch_size)))

        # compute the total number of batch updates that will take place while we are trying to find a good starting lr
        num_batch_updates = epochs * steps_per_epoch

        # derive the lr multiplier based on the ending lr, starting lr, and total number of batch updates
        self.lr_mult = (end_lr / start_lr) ** (1./num_batch_updates)

        # create a temp file path for the model weights and save the weights
        self.weights_file = tempfile.mkstemp()[1]
        self.model.save_weights(self.weights_file)

        # grab the original lr, and set starting lr
        orig_lr = K.get_value(self.model.optimizer.lr)
        K.set_value(self.model.optimizer.lr, start_lr)

        # construct a callback that will be called at the end of each batch, enable us to increase lr as training processes
        callback = LambdaCallback(on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs))

        # check to see if we are using data iterator
        if use_gen:
            self.model.fit_generator(train_X, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=verbose, callbacks=[callback])
        # otherwise, entire training data is already in memory
        else:
            self.model.fit(train_X[0], train_X[1], batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=[callback])

        # restore the original model weights and lr
        self.model.load_weights(self.weights_file)
        K.set_value(self.model.optimizer.lr, orig_lr)

    def plot_loss(self, skip_begin=10, skip_end=1, title=''):
        # grab the lr and losses values to plot
        lrs = self.lrs[skip_begin:-skip_end]
        losses = self.losses[skip_begin:-skip_end]

        # plot the lr vs. loss
        plt.plot(lrs, losses)
        plt.xscale('log')
        plt.xlabel('Learning rate (Log scale)')
        plt.ylabel('Loss')

        if title != '':
            plt.title(title)
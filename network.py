""" From scratch implementation of a multi-layer perceptron"""

import numpy as np

from layer.activations import compute_activation


class MLP:
    """
    Class for creating a multi-layer perceptron. Requires a variable number of layer
    objects and will construct a network from the input layers in order, meaning
    the first layer passed to the class on construction will be the first layer
    when feedforward and backprop get called.

    Inputs (Constructor):
    *layers - Variable number of layer objects from the layer class
    y - One-hot encoded truth vector (must be one-hot)
    learning_rate - Learning rate hyperparameter

    """

    def __init__(self, *layers, y_truth, learning_rate=0.01):
        self.layers = layers
        self.y_truth = y_truth
        self.y_hat = 0
        self.learning_rate = learning_rate
        self.test_loss = 0
        self.test_accuracy = 0
        self.loss = 0
        self.accuracy = 0
        self.avg_loss = 0
        self.avg_accuracy = 0
        self.best_weights_epoch = 0

    def feedforward(self):
        """
        Computes one full feedforward through the network. Starting with the
        first layer, the stand_ard linear calculation is done. If the layer has
        an activation function, the activation is then calculated. Following this,
        the resulting output is set up as the input to the next layer, unless the
        current layer is the output layer.

        Inputs:

        None

        Outputs:

        y_hat - The normalized output of the feedforward calculation.

        """
        for i, _ in enumerate(self.layers):
            self.layers[i].compute_linear()
            if np.char.lower(self.layers[i].activation_function) != "none":
                next_input = self.layers[i].call_activation()
                next_input_extended = self.add_bias_factor(next_input, 0)
            else:
                next_input = self.layers[i].linear_output

            if i != (len(self.layers) - 1):
                self.layers[i + 1].set_input(next_input, next_input_extended)

        self.y_hat = next_input
        return self.y_hat

    def backprop(self, batch_size):
        """
        Computes a single iteration of backpropagation and updates the layer
        weights and biases. Special handling is done to make sure the loss
        derivative gets calculated first, note that the loss derivatives
        are with respect to the unnormalized outputs, not the normalized.

        Inputs:

        batch_size - Just the batch size.

        Outputs:

        None
        """
        for i in reversed(range(len(self.layers))):
            current_layer = self.layers[i]
            loss_function = current_layer.loss_function
            if loss_function != "none":
                if loss_function == "cross_entropy":
                    d_z = current_layer.cross_entropy_derivative(self.y_truth)
                elif loss_function == "hinge":
                    d_z = current_layer.hinge_derivative(self.y_truth)

                d_w = (1 / batch_size) * np.transpose(current_layer.x_extended) @ d_z
            else:
                previous_layer = self.layers[i + 1]

                d_h = d_z @ previous_layer.weights.T

                if current_layer.activation_function == "relu":
                    d_a = current_layer.relu_derivative(current_layer.linear_output)
                elif current_layer.activation_function == "sigmoid":
                    d_a = current_layer.sigmoid_derivative(current_layer.linear_output)

                # Element-wise multiplication
                d_z = np.multiply(d_h, d_a)

                d_w = (1 / batch_size) * np.transpose(current_layer.x_extended) @ d_z

            current_layer.w_extended = current_layer.w_extended - (
                d_w * self.learning_rate
            )

    def cross_entropy_loss(self):
        """
        Computes the Cross-Entropy loss for a multi-class classification
        MLP with an output layer activation that assigns a probability to
        the various possible classifications.

        Outputs:
        Loss - The average loss across the batch where the loss for each example
        is calculated as -log(p), where p is the probability found in y_hat at the
        index corresponding to the correct classification in the one-hot encoded
        test or training label y.
        """
        label_index = np.argwhere(self.y_truth == 1)
        loss = np.sum(-np.log(self.y_hat[label_index[:, 0], label_index[:, 1]]))

        return loss

    def hinge_loss(self):
        """
        Calculates the hinge loss.
        """
        y_i = np.squeeze(np.nonzero(self.y_truth == 1))
        y_hat_unnorm = self.layers[-1].linear_output

        hinge = np.transpose(
            np.maximum(
                0, np.transpose(y_hat_unnorm) - y_hat_unnorm[y_i[0, :], y_i[1, :]] + 1
            )
        )

        # Don't compute the loss for the index into y_hat that corresponds to truth
        hinge[y_i[:, 0], y_i[:, 1]] = 0
        loss = np.sum(hinge)

        return loss

    def compute_accuracy(self):
        """
        Computes the number of classifications that were correct and incorrect.
        The predicted class for the model is simply the class that had the highest
        probability of the outputs, so accuracy does not by itself indicate confidence level.

        Inputs:
        None

        Outputs:
        right_class - Integer, number of predictions that were correct.
        wrong_class - Integer, number of predictions that were incorrect
        """
        wrong_class = 0
        right_class = 0

        for i in range(len(self.y_hat)):
            y_hat_arg = np.argmax(self.y_hat[i, :])
            y_arg = np.argmax(self.y_truth[i, :])

            if y_hat_arg == y_arg:
                right_class = right_class + 1
            else:
                wrong_class = wrong_class + 1

        return right_class, wrong_class

    def compute_loss(self):
        """
        Computes the loss for a given set of predictions. Calls the relevant
        loss function based on the set loss function at class instantiation.
        """
        if self.layers[-1].loss_function == "hinge":
            loss = self.hinge_loss()
        elif self.layers[-1].loss_function == "cross_entropy":
            loss = self.cross_entropy_loss()

        return loss

    def train(
        self,
        epochs,
        num_loops,
        batch_size,
        x_train,
        y_train,
        x_test,
        y_test,
        find_best_weights,
    ):

        """
        The training loop that calls feedforward and backprop for a specified number of epochs.

        Inputs:

        epochs - The number of epochs to train.
        num_loops - This is the floor of the number of examples in the training set
        divided by the batch size. It determines how many times the inner loop needs to run to go
        through all of the examples for a given batch size.

        batch_size - Just the size of the batches.
        x_train - Input training d_ata.
        y_train - Truth values for the training d_ata.
        x_test - Input test d_ata.
        y_test - Truth values for test d_ata
        find_best_weights - Set to 1 if you want to compare against test set during training
        to find the best weights.

        Outputs:
        None
        """
        self.avg_loss = np.zeros(epochs)
        self.avg_accuracy = np.zeros(epochs)
        self.test_loss = np.zeros(epochs)
        self.test_accuracy = np.zeros(epochs)

        for i in range(epochs):

            # Shuffle the dataset with a different seed each loop
            # so that the shuffling isn't the same every epoch.
            np.random.seed(i)
            permutation = np.random.permutation(x_train.shape[0])
            train_x_shuffled = x_train[permutation, :]
            train_y_shuffled = y_train[permutation, :]

            self.loss = np.array(np.zeros(num_loops))
            self.accuracy = np.array(np.zeros(num_loops))

            for j in range(num_loops):
                batch_end = (j + 1) * batch_size
                batch_start = batch_end - batch_size

                # Set the first layer input
                x_train_extended = self.add_bias_factor(
                    train_x_shuffled[batch_start:batch_end, :],
                    0,
                )

                self.layers[0].set_input(
                    train_x_shuffled[batch_start:batch_end, :],
                    x_train_extended,
                )

                self.y_truth = train_y_shuffled[batch_start:batch_end, :]
                self.y_hat = self.feedforward()
                self.backprop(batch_size)

                self.loss[j] = self.compute_loss() / batch_size
                right_class, _ = self.compute_accuracy()
                self.accuracy[j] = (right_class / batch_size) * 100

            self.avg_loss[i] = np.sum(self.loss) / num_loops
            self.avg_accuracy[i] = np.sum(self.accuracy) / num_loops

            if find_best_weights:
                x_test_extended = self.add_bias_factor(x_test, 0)
                self.layers[0].set_input(x_test, x_test_extended)

                self.y_truth = y_test
                self.y_hat = self.feedforward()

                self.test_loss[i] = np.sum(self.compute_loss()) / len(y_test)
                right_class, _ = self.compute_accuracy()
                self.test_accuracy[i] = (right_class / len(y_test)) * 100

                if i == 0:
                    current_best_loss = self.test_loss[i]
                    for j, _ in enumerate(self.layers):
                        self.layers[j].best_w = self.layers[j].w_extended
                        self.best_weights_epoch = i
                else:
                    if self.test_loss[i] < current_best_loss:
                        current_best_loss = self.test_loss[i]
                        for j, _ in enumerate(self.layers):
                            self.layers[j].best_w = self.layers[j].w_extended
                            self.best_weights_epoch = i

    def add_bias_factor(self, matrix, add_to_row):
        """
        This is basically just a utility function so that there isn't a lot
        of redundant code every time extended values need to be added to a matrix.
        I found this clever way of adding rows or columns to matrices on Stack Overflow:
        https://stackoverflow.com/questions/8486294/how-to-add-an-extra-column-to-a-numpy-array
        """
        if add_to_row:
            matrix_extended = np.zeros((matrix.shape[0] + 1, matrix.shape[1]))
            matrix_extended[:-1, :] = matrix
        else:
            matrix_extended = np.ones((matrix.shape[0], matrix.shape[1] + 1))
            matrix_extended[:, :-1] = matrix

        return matrix_extended

    def set_to_best_weights(self):
        """
        Call this to set each layer's weights to the best weights.
        """
        for i, _ in enumerate(self.layers):
            self.layers[i].w_extended = self.layers[i].best_w


class Layer:
    """
    Layer class, used for creating a single layer and handles both the relevant
    information about the layer, such as weight, bias, input, unnormalized output,
    and normalized output, but also handles all of the computations that need to be
    done in that layer.

    Inputs (Constructor):

    activation_function - String with the name of the activation function for the layer.
    These can be: (relu, sigmoid, softmax, none).

    num_inputs - Number of inputs to the layer

    num_outputs - Number of outputs coming from the layer

    loss_function - String specifying the loss function (hinge, softmax, none)
    """

    def __init__(
        self,
        activation_function="none",
        num_inputs=1,
        num_outputs=1,
        loss_function="none",
        init_type="normal",
    ):

        self.activation_function = activation_function
        self.loss_function = loss_function
        self.activation_output = 0
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.y_hat = 0
        self.linear_output = 0
        self.best_w = 0
        self.x_input = 0
        self.x_extended = 0

        self.init_params(init_type)

    def cross_entropy_derivative(self, y_truth):
        """
        Calculates the derivative of cross-entropy loss with respect to
        the unnormalized y_hat. This derivative assumes that the unnormalized y_hat is the
        input to a softmax activation in the output layer, so the derivative
        reduces to dloss/d_z = dloss/dy_hat * dy_hat/d_z, where y_hat is the output
        of softmax(Z) and Z is the unnormalized y_hat.

        Inputs: y, a one-hot encoded vector of truth values

        Outputs: dloss/d_z, see above.
        """

        return self.y_hat - y_truth

    def relu_derivative(self, x_input):
        """
        Computes the relu derivative for an input X, and returns the resulting matrix.
        """
        d_x = np.zeros(x_input.shape)
        d_x[x_input >= 0] = 1

        return d_x

    def sigmoid_derivative(self, x_input):
        """
        Computes the sigmoid derivative for an input y, and returns the resulting matrix.
        Also used for softmax.
        """
        sigmoid = self.sigmoid(x_input)
        sigmoid_derivative = sigmoid * (1 - sigmoid)

        return sigmoid_derivative

    def hinge_derivative(self, y_truth):
        """
        Derivative for hinge loss with respect to the unnormalized output of the final
        layer. The derivative is calculated like this: If the result of the hinge loss
        calculation for some Zi, where i is NOT equal to the index into y that corresponds
        to the truth value, is Zi - Zj + 1, then the derivative is 1. If the result is 0,
        the derivative is 0. For j, where j is the truth index, the derivative is the negative
        of the number of times Zi - Zj + 1 is calculated for a particular output. In other words,
        if the derivative has already been calculated for all indices not j, the derivative of Zj
        is -n where n is the number of times 1 was the derivative for the other outputs Zi.

        The following post was extremely helpful when I was trying to figure out the
        vectorized implementation for this:
        https://mlxai.github.io/2017/01/06/vectorized-implementation-of-svm-loss-and-gradient-update.html

        """
        y_i = np.squeeze(np.nonzero(y_truth == 1))
        hinge = np.transpose(
            np.maximum(
                0,
                np.transpose(self.linear_output)
                - self.linear_output[y_i[0, :], y_i[1, :]]
                + 1,
            )
        )
        hinge[y_i[1, :]] = 0

        # The derivative at indices where the hinge calculation did not return 0 is 1,
        # 0 elsewhere (except for the truth index)
        hinge[hinge > 0] = 1

        # Make sure value at truth index is 0
        hinge[y_i[0, :], y_i[1, :]] = 0

        # The derivative at the truth index can now be expressed as the
        # negative of the sum of the number of ones
        grad_at_yi = -np.sum(hinge, axis=1)
        hinge[y_i[0, :], y_i[1, :]] = grad_at_yi

        hinge_derivative = hinge
        return hinge_derivative

    def compute_linear(self):
        """
        Standard linear computation for a layer, note that it is XW + b because
        my batch size is the row instead of the column, so it swaps the order
        of the multiplication from what we did in class. I did this because it's
        the way the d_ata is when it first gets loaded, so no reason to change it.
        """
        self.linear_output = self.x_extended @ self.w_extended

        return self.linear_output

    def call_activation(self):
        """ Calls the proper activation function for a layer depending on what that
        layer's specified activation is."""

        self.activation_output = compute_activation(self.linear_output, self.activation_function)

        if self.activation_function == "softmax":
            self.y_hat = self.activation_output

        return self.activation_output

    def init_params(self, init_type="normal"):
        """
        Initializes the weights and biases.

        Inputs:

        init_type - String, zero or normal
        """

        if init_type == "normal":
            self.weights = np.random.normal(
                0, 0.01, (self.num_inputs, self.num_outputs)
            )
        else:
            self.weights = np.zeros((self.num_inputs, self.num_outputs))

        # Set the extended weights after initialization so that the bias is set to 0
        self.w_extended = np.zeros((self.weights.shape[0] + 1, self.weights.shape[1]))
        self.w_extended[:-1, :] = self.weights

    def set_input(self, x_input, x_extended):
        """Set the input values for the layer"""
        self.x_input = x_input
        self.x_extended = x_extended

    def set_activation_function(self, new_function):
        """Set the activation function for the layer"""
        self.activation_function = new_function

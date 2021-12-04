from tensorflow.math import reduce_sum, count_nonzero
from tensorflow import squeeze, reduce_mean
from tensorflow.keras.backend import ctc_batch_cost

class Loss_Calculation:

    def ctc_loss_lambda_func(self, y_true, y_pred):
        """Function for computing the CTC loss"""

        if len(y_true.shape) > 2:
            y_true = squeeze(y_true)

        input_length = reduce_sum(y_pred, axis=-1, keepdims=False)
        input_length = reduce_sum(input_length, axis=-1, keepdims=True)


        label_length = count_nonzero(y_true, axis=-1, keepdims=True, dtype="int64")

        loss = ctc_batch_cost(y_true, y_pred, input_length, label_length)

        loss = reduce_mean(loss)

        return loss

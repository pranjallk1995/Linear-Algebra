import json
import tensorflow as tf

from tensorflow.keras.layers import Dense

# custom loss
def custom_loss(y_true, y_pred, extra_parameter=10):
    return extra_parameter*(y_true - y_pred)

# make model
model = tf.keras.Sequential()
model.add(
    Dense(
        units=1, activation='sigmoid', input_shape=(2,),                                           # Notice the input shape
        kernel_initializer='glorot_normal', bias_initializer='glorot_normal'
    )
)

# check model
print(model.summary())

# compile model
model.compile(loss=custom_loss)

# train model
x = tf.convert_to_tensor([[1., 1.]])                                                              # Notice the x shape compared to x
y = tf.convert_to_tensor([0.])

model.fit(x, y)

# check:
weights = tf.convert_to_tensor(model.layers[0].get_weights()[0])
bias = tf.convert_to_tensor(model.layers[0].get_weights()[1])

linear_composition = weights[0]*x[0][0] + weights[1]*x[0][1] + bias
activation_output = tf.sigmoid(linear_composition)

# match this with the loss shown during fit
print(f"calculated loss value: {10*(y - activation_output)}")

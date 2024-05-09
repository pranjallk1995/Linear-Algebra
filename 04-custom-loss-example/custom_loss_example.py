import tensorflow as tf

from tensorflow.keras.layers import Dense

# custom loss
def custom_loss(y_true, y_pred, extra_parameter=10):
    return extra_parameter*(y_true - y_pred)

# make model
model = tf.keras.Sequential()
model.add(Dense(1, activation='sigmoid', input_shape=(1,)))

# check model
print(model.summary())

# compile model
model.compile(loss=custom_loss)

# train model
x = tf.convert_to_tensor([1.])
y = tf.convert_to_tensor([0.])

model.fit(x, y)

# check:
weight = tf.convert_to_tensor(model.layers[0].get_weights()[0])
bias = tf.convert_to_tensor(model.layers[0].get_weights()[1])

linear_composition = weight[0]*x + bias
activation_output = tf.sigmoid(linear_composition)

# match this with the loss show during fit
print(f"calculated loss value: {10*(y - activation_output)}")

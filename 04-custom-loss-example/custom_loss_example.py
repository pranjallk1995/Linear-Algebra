import tensorflow as tf

from tensorflow.keras.layers import Dense

# custom loss
def custom_loss(y_true, y_pred, extra_parameter=10):
    return extra_parameter*(y_true - y_pred)

# make model
model = tf.keras.Sequential()
model.add(
    Dense(
        units=2, activation='sigmoid', input_shape=(3,),                                                        # notice the input shape as compare to feature space
        kernel_initializer='glorot_normal', bias_initializer='glorot_normal'
    )
)

# check model
print(model.summary())

# compile model
model.compile(loss=custom_loss)

# train model
x = tf.convert_to_tensor([[1., 1., 2.], [2., 2., 1.]])
y = tf.convert_to_tensor([[0., 1.], [1., 0.]])

model.fit(x, y)

# check:
weights = tf.convert_to_tensor(model.layers[0].get_weights()[0])                                                
bias = tf.convert_to_tensor(model.layers[0].get_weights()[1])

network_output = []
for feature_vector in x:
    feature_output = []
    for neuron in range(2):
        linear_combination = tf.reduce_sum(weights[:, neuron]*feature_vector) + bias[neuron]
        activation_ouptut = tf.sigmoid(linear_combination)
        feature_output.append(activation_ouptut)
    network_output.append(feature_output)

calculated_ypred = tf.convert_to_tensor(network_output)

# match this with the loss shown during fit
print(f"calculated loss value: {tf.reduce_mean(10*(y - calculated_ypred))}")                                    # notice that mean is take by default.

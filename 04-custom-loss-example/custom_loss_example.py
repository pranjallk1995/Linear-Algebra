import tensorflow as tf

from tensorflow.keras.layers import Dense

# custom loss
def custom_loss(y_true, y_pred, extra_parameter=10):
    return extra_parameter*(y_true - y_pred)

# make model
model = tf.keras.Sequential()
model.add(
    Dense(
        units=1, activation='sigmoid', input_shape=(3,),                                               # notice the input shape based on the feature space below
        kernel_initializer='glorot_normal', bias_initializer='glorot_normal'
    )
)

# check model
print(model.summary())

# compile model
model.compile(loss=custom_loss)

# train model
x = tf.convert_to_tensor([[1., 1., 2.], [2., 2., 1.]])                                                 # notice the feature space shape
y = tf.convert_to_tensor([[0.], [1.]])

model.fit(x, y)

# check:
weights = tf.convert_to_tensor(model.layers[0].get_weights()[0])
bias = tf.convert_to_tensor(model.layers[0].get_weights()[1])

""" input 1: """
linear_composition_1 = weights[0]*x[0][0] + weights[1]*x[0][1] + weights[2]*x[0][2] + bias
activation_output_1 = tf.sigmoid(linear_composition_1)

""" input 2: """
linear_composition_2 = weights[0]*x[1][0] + weights[1]*x[1][1] + weights[2]*x[1][2] + bias
activation_output_2 = tf.sigmoid(linear_composition_2)

calculated_ypred = tf.convert_to_tensor([activation_output_1, activation_output_2])

# match this with the loss shown during fit
print(f"calculated loss value: {tf.reduce_mean(10*(y - calculated_ypred))}")                            # notice that mean is taken by default

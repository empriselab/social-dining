from tcn import TCN, tcn_full_summary
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import tensorflow as tf
from keras import backend as K
# if time_steps > tcn_layer.receptive_field, then we should not
# be able to solve this task.
batch_size, time_steps, input_dim = None, 20, 1

# config = tf.compat.v1.ConfigProto(device_count={"CPU": 7}, intra_op_parallelism_threads=4,
#                         inter_op_parallelism_threads=4, 
# )
# K.set_session(tf.compat.v1.Session(config=config))

# config = tf.ConfigProto(intra_op_parallelism_threads=4,
#                         inter_op_parallelism_threads=4, 
#                         allow_soft_placement=True,
#                         device_count = {'CPU' : 4,
#                                         'GPU' : 1}
#                        )

# session = tf.Session(config=config)
# K.set_session(session)


def get_x_y(size=1000):
    import numpy as np
    pos_indices = np.random.choice(size, size=int(size // 2), replace=False)
    x_train = np.zeros(shape=(size, time_steps, 1))
    y_train = np.zeros(shape=(size, 1))
    x_train[pos_indices, 0] = 1.0  # we introduce the target in the first timestep of the sequence.
    y_train[pos_indices, 0] = 1.0  # the task is to see if the TCN can go back in time to find it.
    return x_train, y_train


tcn_layer = TCN(input_shape=(time_steps, input_dim))
# The receptive field tells you how far the model can see in terms of timesteps.
print('Receptive field size =', tcn_layer.receptive_field)

m = Sequential([
    tcn_layer,
    Dense(1)
])

print("Compiling...")
m.compile(optimizer='adam', loss='mse')
print("Done compiling")
tcn_full_summary(m, expand_residual_blocks=False)
print('Getting xy')
x, y = get_x_y()
print("fitting...")
print(x.shape, y.shape)
m.fit(x, y, epochs=10, validation_split=0.2, use_multiprocessing=True)
#%%
import os
from time import time
import joblib
import random
from CustomModels import deeponet as DON
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import optimizers, losses, models, layers


# tf.config.run_functions_eagerly(
#     run_eagerly = True
# )



SEED = 24
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#   tf.config.experimental.set_memory_growth(gpu, True)

# tf.config.set_visible_devices(gpus, 'GPU')



#%%

def normscaler(df):
    X = df.values[:, :3]
    y = df.values[:, -1]
    
    a = np.std(X, axis=0)
    b = np.std(y, axis=0)
    
    ma = np.mean(X, axis=0)
    mb = np.mean(y, axis=0)
    
    X = (X - ma) / a
    y = (y - mb) / b
    
    params = [a, b, ma, mb]
    return (X, y, params)

def scale_df(df, params):
    X = df.values[:, :3]
    y = df.values[:, -1]
    
    X = (X - params[2]) / params[0]
    y = (y - params[3]) / params[1]
    
    return(X, y)
#%%

model_name = "DON"
PATH = os.getcwd()

if not os.path.exists(f'./models/{model_name}'):
  os.makedirs(f'./models/{model_name}')

sdir = f'./models/{model_name}/'
df = joblib.load(os.path.join(PATH, 'DATA', 'data'))

df[0].columns = ['lambda2']
df[1].columns = ['Ux', 'Uy', 'Uz']

data = pd.concat([df[1], df[0]], axis=1)

#split the data to the train and test portions
df_test = data.sample(frac=0.2, random_state = SEED)
df_test_idx = df_test.index
df_train = data.drop(df_test_idx)

#
X_train, y_train, params= normscaler(df_train)
X_test, y_test = scale_df(df_test, params)

grid = np.load(os.path.join(PATH, 'DATA', 'grids.npy'), allow_pickle=True)

# %%



def gen_models(nf, nv, act, nn, nl, n_out):
    inp = layers.Input((nf,))
    x = layers.Dense(nn, activation = act)(inp)
    for i in range(nl - 1):
        x = layers.Dense(nn, activation = act)(x)
        
    outs = []
    for i in range(n_out):
        outs.append(layers.Dense(nv)(x))
        
    model = models.Model(inp, outs)
    return model

m = 3
r = 10

act = tf.keras.activations.swish
branch = gen_models(m, r, act, 64, 4, 1)
print(branch.summary())

trunk = gen_models(3, r, act, 128, 4, 3)
print(trunk.summary())

#%%
n_batches = 100
batch_size = int(len(X_train) / n_batches)

initial_learning_rate = 1e-3
lr_schedule = optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000 * n_batches,
    decay_rate=0.1,
    staircase=False)

model = DON(branch, trunk)
model.compile_grid_modes(grid)
model.compile(
    optimizer = optimizers.Adam(learning_rate=lr_schedule),
    loss = losses.MeanSquaredError()
    )
start_time = time()
hist = model.fit(X_train, y_train, 
          epochs=4000, 
          batch_size = batch_size, 
          validation_data=(X_test, y_test), 
          verbose = 2)
end_time = time()
cp_time = start_time - end_time

model.save(sdir + 'model', save_format = 'tf')
hist = hist.history

hist = np.array(hist)
np.savez_compressed(sdir + 'res', hist = hist, cp_time = cp_time)
#joblib.dump(pp, sdir + 'pp')
#%%
y_pred = model.predict(X_test)
#%%
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape = (3,)),
#     #tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(16, activation='relu'),
#     #tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(8, activation='relu'),
#     #tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(1)
#     ])
# lr_schedule = optimizers.schedules.ExponentialDecay(
#     initial_learning_rate = 1e-3,
#     decay_steps=100 * n_batches,
#     decay_rate=0.1,
#     staircase=False)
# model.compile(
#     optimizer='adam',
#     loss = tf.keras.losses.MeanSquaredError(),
#     )

# earlyStopCallBack = callbacks.EarlyStopping(monitor='loss',
#                                             patience=3,
#                                             start_from_epoch = 100,
#                                        )

# history = model.fit(X_train,
#                     y_train,
#                     validation_data=(X_test, y_test),
#                     epochs=500,
#                     batch_size=batch_size,
#                     callbacks=[earlyStopCallBack]
#                     )
# %%
# import matplotlib.pyplot as plt

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()


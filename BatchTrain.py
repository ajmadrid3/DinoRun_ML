import numpy as np
from GameParameters import BATCH, GAMMA
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD , Adam
from keras.callbacks import TensorBoard

def trainBatch(minibatch):
    inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))
    targets = np.zeros
    loss = 0

    for i in range(0, len(minibatch)):
        state_t = minibatch[i][0]   # 4D stack of images
        action_t = minibatch[i][1]  # Action index
        reward_t = minibatch[i][2]  # Reward at state_t due to action_t
        state_t1 = minibatch[i][3]  # Next state
        terminal = minibatch[i][4]  # Whether the agent died or survied from the action
        inputs[i:i + 1] = state_t
        targets[i] = model.predict(state_t)
        Q_sa = model.predict(state_t1)
        if terminal:
            targets[i, action_t] = reward_t
        else:
            targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)

        loss += model.train_on_batch(inputs, targets)
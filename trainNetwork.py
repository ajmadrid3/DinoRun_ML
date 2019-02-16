import numpy as np
import random 
import time
from collections import deque
from buildModel import ACTIONS
from GameParameters import OBSERVATION, INITIAL_EPSILON, FINAL_EPSILON, EXPLORE, REPLAY_MEMORY, BATCH, GAMMA
from BatchTrain import trainBatch


def trainNetwork(model, game_state):
    # Stores the previous observations in replay memory.
    D = deque()
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1 #0 -> Do Nothing, #1 -> Jump

    x_t, r_0, terminal = game_state.get_state(do_nothing) 
    s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2).reshape(1, 20, 40, 4) # Stack of four images as placeholder for input

    OBSERVE = OBSERVATION
    epsilon = INITIAL_EPSILON
    t = 0

    while(True): 

        loss = 0
        Q_sa = 0
        action_index = 0
        r_t = 0 #Reward at t
        a_t = np.zeros([ACTIONS]) #Action at t

        #chooses an action 
        if random.random() <= epsilon: #radomly chooses
            action_index = random.randrange(ACTIONS)
            a_t[action_index] = 1
        else: #Predicts the action
            q = model.predict(s_t)
            max_q = np.argmax(q)
            action_index = max_q
            a_t[action_index] = 1

        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        x_t1, x_r, terminal = game_state.get_state(a_t)
        last_time = time.time()
        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis = 3)

        #Stores the transition
        D.append((s_t, action_index, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        if t > OBSERVE:
            minibatch = random.sample(D, BATCH)
            inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))
            targets = np.zeros((inputs.shape[0], ACTIONS))
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


        s_t = s_t1
        t = t + 1

        print("TimeStep ",t, "/ Epsilon ", epsilon, "/ Action ", action_index, "/ Reward ", r_t, "/ Max Q ", np.max(Q_sa), "/ Loss ", loss)
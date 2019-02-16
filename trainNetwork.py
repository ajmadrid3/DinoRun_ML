import numpy as np
import random as random
from collections import deque

def trainNetwork(model, game_state):
    # Stores the previous observations in replay memory.
    D = deque()
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1 #0 -> Do Nothing, #1 -> Jump

    x_t, r_0, terminal = game_state.get_state(do_nothing) 
    s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2).reshape(1, 20, 40, 4) # Stack of four images as placeholder for input

    OBSERVE = OBSERVATION
    epsilon = INITAL_EPSILION
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
            epsilon -= (INITAL_EPSILION - FINAL_EPSILON) / EXPLORE

        x_t1, x_r, terminal = game_state.get_state(a_t)
        last_time = time.time()
        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis = 3)

        #Stores the transition
        D.append((s_t, action_index, r_t, s_t1, terminal))
        D.popleft() if len(D) > REPLAY_MEMORY

        trainBatch(random.sample(D, Batch)) if t > OBSERVE
        s_t = s_t1
        t = t++

        print("TimeStep ",t, "/ Epsilon ", epsilon, "/ Action ", action_index, "/ Reward ", r_t)
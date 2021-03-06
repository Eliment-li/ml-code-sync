from tianshou.data import Batch, ReplayBuffer

def init():
    # a buffer is initialised with its maxsize set to 10 (older data will be discarded if more data flow in).
    print("========================================")
    buf = ReplayBuffer(size=10)
    print(buf)
    print("maxsize: {}, data length: {}".format(buf.maxsize, len(buf)))

    # add 3 steps of data into ReplayBuffer sequentially
    print("========================================")
    for i in range(3):
        buf.add(Batch(obs=i, act=i, rew=i, done=0, obs_next=i + 1, info={}))
    print(buf)
    print("maxsize: {}, data length: {}".format(buf.maxsize, len(buf)))

    # add another 10 steps of data into ReplayBuffer sequentially
    print("========================================")
    for i in range(3, 13):
        buf.add(Batch(obs=i, act=i, rew=i, done=0, obs_next=i + 1, info={}))
    print(buf)
    print("maxsize: {}, data length: {}".format(buf.maxsize, len(buf)))

    print(buf[-1])
    print(buf[-3:])
    import pickle
    _buf = pickle.loads(pickle.dumps(buf))

    '''Data sampling'''
    buf.sample(5)
buf = ReplayBuffer(size=10)
def TrajectoryTracking():
    from numpy import False_

    # Add the first trajectory (length is 3) into ReplayBuffer
    print("========================================")
    for i in range(3):
        result = buf.add(Batch(obs=i, act=i, rew=i, done=True if i == 2 else False, obs_next=i + 1, info={}))
        print(result)
    print(buf)
    print("maxsize: {}, data length: {}".format(buf.maxsize, len(buf)))
    # Add the second trajectory (length is 5) into ReplayBuffer
    print("========================================")
    for i in range(3, 8):
        result = buf.add(Batch(obs=i, act=i, rew=i, done=True if i == 7 else False, obs_next=i + 1, info={}))
        print(result)
    print(buf)
    print("maxsize: {}, data length: {}".format(buf.maxsize, len(buf)))
    # Add the third trajectory (length is 5, still not finished) into ReplayBuffer
    print("========================================")
    for i in range(8, 13):
        result = buf.add(Batch(obs=i, act=i, rew=i, done=False, obs_next=i + 1, info={}))
        print(result)
    print(buf)
    print("maxsize: {}, data length: {}".format(buf.maxsize, len(buf)))

    '''ReplayBuffer.add() ????????????(current_index, episode_reward, episode_length, episode_start_index]
    episode_reward ??? episode_length ????????? trajectory ???????????????????????????
    '''

#run after  TrajectoryTracking()
def index():
    print(buf)
    data = buf[6]
    print(data)

    # Search for the previous index of index "6"
    '''
    ?????????Trajectory????????? ????????? 4-7 ????????? Trajectory????????? 7???next ????????????
    '''
    now_index = 6
    while True:
        prev_index = buf.prev(now_index)
        print(prev_index)
        if prev_index == now_index:
            break
        else:
            now_index = prev_index

    # next step of indexes [4,5,6,7,8,9] are:
    print(buf.next([4, 5, 6, 7, 8, 9]))

    #?????? labele "done: False" ?????????, ?????????????????? Trajectory ?????????
    print(buf.unfinished_index())

if __name__ == '__main__':
    init()
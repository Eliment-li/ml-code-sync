import numpy as np
import torch
from tianshou.data import Batch


data = Batch(a=4, b=[5, 5], c='2312312', d=('a', -2, -3))
print(data)
print(data.b)

def Initialisation():
    # converted from a python library
    print("1========================================")
    batch1 = Batch({'a': [4, 4], 'b': (5, 5)})
    print(batch1)

    # initialisation of batch2 is equivalent to batch1
    print("2========================================")
    batch2 = Batch(a=[4, 4], b=(5, 5))
    print(batch2)

    # the dictionary can be nested, and it will be turned into a nested Batch
    print("3========================================")
    data = {
        'action': np.array([1.0, 2.0, 3.0]),
        'reward': 3.66,
        'obs': {
            "rgb_obs": np.zeros((3, 3)),
            "flatten_obs": np.ones(5),
        },
    }

    batch3 = Batch(data, extra="extra_string")
    print(batch3)
    # batch3.obs is also a Batch
    print(type(batch3.obs))
    print(batch3.obs.rgb_obs)

    # a list of dictionary/Batch will automatically be concatenated/stacked, providing convenience if you
    # want to use parallelized environments to collect data.
    print("4========================================")
    batch4 = Batch([data] * 3)
    print(batch4)
    print(batch4.obs.rgb_obs.shape)

def Access():
    batch1 = Batch({'a': [4, 4], 'b': (5, 5)})
    print(batch1)
    # add or delete key-value pair in batch1
    print("1========================================")
    batch1.c = Batch(c1=np.arange(3), c2=False)
    del batch1.a
    print(batch1)

    # access value by key
    print("2========================================")
    assert batch1["c"] is batch1.c
    print("c" in batch1)

    # traverse the Batch
    print("3========================================")
    for key, value in batch1.items():
        print(str(key) + ": " + str(value))

def IndexingAndSlicing():
    # Let us suppose we've got 4 environments, each returns a step of data
    step_datas = [
        {
            "act": np.random.randint(10),
            "rew": 0.0,
            "obs": np.ones((3, 3)),
            "info": {"done": np.random.choice(2), "failed": False},
        } for _ in range(4)
    ]
    batch = Batch(step_datas)
    print(batch)
    print(batch.shape)

    # advanced indexing is supported, if we only want to select data in a given set of environments
    print("========================================")
    #将每个key对应的value[0]取出
    print(batch[0])
    print(batch[[0, 3]])

    # slicing is also supported
    print("========================================")
    print(batch[-2:])
def AggregationAndSplitting():
    # concat batches with compatible keys
    # try incompatible keys yourself if you feel curious
    print("========================================")
    b1 = Batch(a=[{'b': np.float64(1.0), 'd': Batch(e=np.array(3.0))}])
    b2 = Batch(a=[{'b': np.float64(4.0), 'd': {'e': np.array(6.0)}}])
    b12_cat_out = Batch.cat([b1, b2])
    print(b1)
    print(b2)
    print(b12_cat_out)

    # stack batches with compatible keys
    # try incompatible keys yourself if you feel curious
    print("========================================")
    b3 = Batch(a=np.zeros((3, 2)), b=np.ones((2, 3)), c=Batch(d=[[1], [2]]))
    b4 = Batch(a=np.ones((3, 2)), b=np.ones((2, 3)), c=Batch(d=[[0], [3]]))
    b34_stack = Batch.stack((b3, b4), axis=1)
    print(b3)
    print(b4)
    print(b34_stack)

    # split the batch into small batches of size 1, breaking the order of the data
    print("========================================")
    print(type(b34_stack.split(1)))
    print(list(b34_stack.split(1, shuffle=True)))
def Converting():
    import torch
    batch1 = Batch(a=np.arange(2), b=torch.zeros((2, 2)))
    batch2 = Batch(a=np.arange(2), b=torch.ones((2, 2)))
    batch_cat = Batch.cat([batch1, batch2, batch1])
    print(batch_cat)
    #batch 同时支持 tensor 和 numpy 格式的数据
    batch_cat.to_numpy()
    print(batch_cat)
    batch_cat.to_torch()
    print(batch_cat)
def save():
    import pickle
    batch = Batch(obs=Batch(a=0.0, c=torch.Tensor([1.0, 2.0])), np=np.zeros([3, 4]))
    batch_pk = pickle.loads(pickle.dumps(batch))
    print(batch_pk)
if __name__ == '__main__':
    save()
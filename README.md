# Deep Networks with Stochastic Depth
Keras implementation for "Deep Networks with Stochastic Depth" http://arxiv.org/abs/1603.09382

Original code is at https://github.com/yueatsprograms/Stochastic_Depth.


## Usage

1. Install [Theano](https://github.com/Theano/Theano) following its instruction.
2. Install [Keras](https://github.com/fchollet/keras) (I use new API at `keras-1` branch)

```
$ cd keras
$ git checkout keras-1
$ python setup.py install
```

3. Just run `python train.py`


## Known Issues

- Error related to maximum recursion depth
  - When the network is deep, there happens error saying it reaches to maximum recursion depth.
  - You can resolve this issue by using `sys.setrecursionlimit(max_recursion_depth)`. You should increase `max_recursion_depth` until you get no error (Increasing this value might cause segmentation fault if you don't have enough memory).


## Results

- Number of layers == 50
- (other configs are same as `train.py`)

![results](https://cloud.githubusercontent.com/assets/10726958/14477064/904b573e-0146-11e6-865d-99fbd060486e.png)

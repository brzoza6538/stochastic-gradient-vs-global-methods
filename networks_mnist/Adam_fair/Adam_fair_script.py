
from comparison import *

from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784')
images = mnist.data.to_numpy(dtype=np.float32)[:POP_SIZE]
labels = mnist.target.to_numpy(dtype=np.int64)[:POP_SIZE]

gather_data(partial(run_adam_fair_net, images=images, labels=labels), "adam_fair")

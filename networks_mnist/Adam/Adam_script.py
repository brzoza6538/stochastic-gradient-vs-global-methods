
from comparison import *

from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784')
images = mnist.data.to_numpy(dtype=np.float32)
labels = mnist.target.to_numpy(dtype=np.int64)

gather_data(partial(run_adam_net, images=images, labels=labels), "adam_net")

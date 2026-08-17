
from comparison import *

from sklearn import datasets

iris = datasets.load_iris()

images, labels = iris.data, iris.target.astype(int)

gather_data(partial(run_adam_net, images=images, labels=labels), "adam_net_aclosstim_tanh")

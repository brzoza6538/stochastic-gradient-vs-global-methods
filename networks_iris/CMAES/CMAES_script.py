
from comparison import *

from sklearn import datasets

iris = datasets.load_iris()

images, labels = iris.data, iris.target.astype(int)

gather_data(partial(run_cmaes_net, images=images, labels=labels), "cmaes_NEW_1")
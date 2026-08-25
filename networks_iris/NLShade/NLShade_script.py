
from comparison import *

from sklearn import datasets

iris = datasets.load_iris()

images, labels = iris.data, iris.target.astype(int)

gather_data(partial(run_nlshade_net, images=images, labels=labels), "nl_NEW_12") 
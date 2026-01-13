
from comparison import *



mnist = fetch_openml('mnist_784')
images, labels = mnist.data, mnist.target.astype(int)

gather_data(partial(run_adam_net, images=images, labels=labels), "adam_net")
 
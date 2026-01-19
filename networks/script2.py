from comparison import *

mnist = fetch_openml('mnist_784')
images, labels = mnist.data, mnist.target.astype(int)

gather_data(partial(run_cmaes_net, images=images, labels=labels), "cmaes_net_batch_20") 
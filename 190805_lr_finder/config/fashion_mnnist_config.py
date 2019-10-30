from os.path import sep

# initialize the list of class label names
CLASSES = ["top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]

# define minimum and maximum lr, batch size, step size, CLR policy, and number of epochs
MIN_LR = 1e-6
MAX_LR = 1e-2
BATCH_SIZE = 64
STEP_SIZE = 8
NUM_EPOCHS = 48
CLR_METHOD = 'triangular'

# define the path to the output training history plot and cyclical lr plot
TRAINING_PLOT_PATH = sep.join(['output', 'fashion_mnist_training_plot.png'])
CLR_PLOT_PATH = sep.join(['output', 'fashion_mnist_clr_plot.png'])
LRFINND_PLOT_PATH = sep.join(['output', 'fashion_mnist_lrfind_plot.png'])
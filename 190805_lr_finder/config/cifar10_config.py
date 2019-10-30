from os.path import sep

# initialize the list of class label names
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# define minimum and maximum lr, batch size, step size, CLR policy, and number of epochs
MIN_LR = 1e-7
MAX_LR = 1e-2
BATCH_SIZE = 64
STEP_SIZE = 8
NUM_EPOCHS = 96
CLR_METHOD = 'triangular'
# NUM_CLR_CYCLES = 96 / (8 * 2) = 6

# define the path to the output training history plot and cyclical lr plot
TRAINING_PLOT_PATH = sep.join(['output', 'cifar10_training_plot.png'])
CLR_PLOT_PATH = sep.join(['output', 'cifar10_clr_plot.png'])
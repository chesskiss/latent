from dynamax.hidden_markov_model import GaussianHMM

TRUE = true = True
FALSE = false = False

DEBUG = False

NUM_TRAIN_BATCHS  = 3
NUM_TEST_BATCHS    = 1

NUM_EPOCHS          = 3 if DEBUG else 20 #Increase the pochs before addin , num_iters=NUM_EPOCHS to fit_em or num_epochs=... for fit_sgd
ITER                = 200 if DEBUG else 250 #Num of iterations per epoch
NUM_TIMESTEPS       = 5000 if DEBUG else 100
NUM_TRIALS          = 10 if DEBUG else 100
STUDENTS_NUM        = 1 if DEBUG else 2 # ring = true will double the amount

SGD = DEBUG

'HMM Type and settings'
EMISSION_DIM    = 5
TRUE_NUM_STATES = 10
MIN_S_STATE     = TRUE_NUM_STATES + 0
MAX_S_STATE     = TRUE_NUM_STATES + 0
epsilon         = 0.01
HMM = GaussianHMM(TRUE_NUM_STATES, EMISSION_DIM)
# S_KEYS  = ['0', '1', '00', '01', '11'] #TODO
S_KEYS  = [i for i in range(NUM_EPOCHS)] 

LEARNING_RATE = 1e-1 # 1e-3 is the default step size

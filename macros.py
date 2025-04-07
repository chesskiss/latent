from dynamax.hidden_markov_model import GaussianHMM

TRUE = true = True
FALSE = false = False

DEBUG   = false
VERBOSE = false
SHORT_RUN = DEBUG or true

SGD = false


NUM_TRAIN_BATCHS  = 3
NUM_TEST_BATCHS    = 1

NUM_EPOCHS      = 10 if SHORT_RUN else 20 #Increase the pochs before addin , num_iters=NUM_EPOCHS to fit_em or num_epochs=... for fit_sgd
ITER            = 15 if SHORT_RUN else 30 #Num of iterations per epoch
NUM_TIMESTEPS   = 50 if SHORT_RUN else 50
NUM_TRIALS      = 10 if SHORT_RUN else 100
STUDENTS_NUM    = 3 if SHORT_RUN else 2 # ring = true will double the amount
RING            = true

MULTI_TEACHERS_FIT = false

'HMM Type and settings'
EMISSION_DIM    = 5
TRUE_NUM_STATES = 10
MIN_S_STATE     = TRUE_NUM_STATES + 0
MAX_S_STATE     = TRUE_NUM_STATES + 0
epsilon         = 0.01
HMM = GaussianHMM(TRUE_NUM_STATES, EMISSION_DIM)
# S_KEYS  = ['0', '1', '00', '01', '11'] #TODO
S_KEYS  = [i for i in range(NUM_EPOCHS)] 

LEARNING_RATE = 1e-3 # 1e-3 is the default step size

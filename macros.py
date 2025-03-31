from dynamax.hidden_markov_model import GaussianHMM

NUM_TRAIN_BATCHS  = 3
NUM_TEST_BATCHS    = 1

NUM_EPOCHS          = 20 #3 #Increase the pochs before addin , num_iters=NUM_EPOCHS to fit_em or num_epochs=... for fit_sgd
ITER                = 250 #Num of iterations per epoch
NUM_TIMESTEPS       = 100
NUM_TRIALS          = 1000
STUDENTS_NUM        = 2 #1 # ring = true will double the amount

SGD = True

'HMM Type and settings'
EMISSION_DIM    = 5
TRUE_NUM_STATES = 10
MIN_S_STATE     = TRUE_NUM_STATES + 0
MAX_S_STATE     = TRUE_NUM_STATES + 0
epsilon         = 0.01
HMM = GaussianHMM(TRUE_NUM_STATES, EMISSION_DIM)
# S_KEYS  = ['0', '1', '00', '01', '11'] #TODO
S_KEYS  = [i for i in range(NUM_EPOCHS)] 

LEARNING_RATE = 1e-5

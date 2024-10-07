import subprocess
import sys
import scipy.stats
import numpy as np
import pandas as pd
# from IPython.display import display

try: #TODO repeat for all libs? 
    import dynamax
except ModuleNotFoundError:
    print('installing dynamax')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'dynamax[notebooks]'])
    import dynamax

from functools import partial

from jax import vmap
import jax.numpy as jnp
import jax.random as jr
from scipy.stats import multivariate_normal

from sklearn.mixture import GaussianMixture 
from dynamax.hidden_markov_model import GaussianHMM
from dynamax.hidden_markov_model import DiagonalGaussianHMM
from dynamax.hidden_markov_model import SphericalGaussianHMM
from dynamax.hidden_markov_model import SharedCovarianceGaussianHMM

from visualize import *




#TODO normalize rows/columns only, depending on j/i
def normalize(A):
    A /= A.sum(axis=1, keepdims=True) #Normalize rows
    A = A.at[-1].set(1 - jnp.sum(A[:-1], axis=0)) #last row is defined
    A = A.at[:, -1].set(1 - jnp.sum(A[:, :-1], axis=1)) #last column is defined
    return A


def perturbation(perturbation_num, epsilon, initial_probs, transition_matrix, emissions_means, emissions_cov):
    key = jr.PRNGKey(perturbation_num)
    np.random.seed(perturbation_num)

    initial_probs += jr.uniform(key, shape=(TRUE_NUM_STATES,))
    initial_probs = initial_probs / jnp.sum(initial_probs)
    
    rn  = lambda x : jr.uniform(key, minval = -x, maxval = x)
    p   = rn(epsilon) * jnp.eye(TRUE_NUM_STATES) \
        + rn(epsilon) * jnp.roll(jnp.eye(TRUE_NUM_STATES), 1, axis=1) \
        + rn(epsilon) / TRUE_NUM_STATES
    transition_matrix   += p
    transition_matrix   = jnp.clip(transition_matrix, 0, None) # Ensures non-negativity
    transition_matrix   = normalize(transition_matrix)

    emissions_means += np.random.normal(-epsilon, epsilon, emissions_means.shape) 
    emissions_cov   += np.random.normal(0, epsilon) * jnp.eye(EMISSION_DIM)[None, :, :]
    emissions_cov   += np.random.normal(-epsilon, epsilon) * jnp.eye(EMISSION_DIM)[None, :, :]
    emissions_cov   = (emissions_cov + np.swapaxes(emissions_cov, -2, -1)) / 2 #Create PSD part 1 : ( A + A.T )/ 2
    # Create PSD part 2 : Make diagonal absolute value:
    abs_diags = jnp.abs(jnp.diagonal(emissions_cov, axis1=-2, axis2=-1))
    mask = jnp.eye(emissions_cov.shape[-1]).astype(bool) #Mask 
    emissions_cov = jnp.where(mask, abs_diags[..., jnp.newaxis], emissions_cov) 

    # print(emissions_cov)
    return initial_probs, transition_matrix, emissions_means, emissions_cov

    
def initial(): 
    'Specify initial parameters of the HMM' # TODO change to if Model is not set. if set - transition matrix = model.transition, etc.
    key = jr.PRNGKey(0)  # Random seed
    initial_probs = jr.uniform(key, shape=(TRUE_NUM_STATES,),  minval=0)
    initial_probs = initial_probs / jnp.sum(initial_probs)

    # initial_probs = jnp.ones(TRUE_NUM_STATES) / TRUE_NUM_STATES

    rn = lambda x : jr.uniform(key, minval = -x, maxval = x)
    transition_matrix   = (0.80 + rn(epsilon)) * jnp.eye(TRUE_NUM_STATES) \
                        + (0.15 + rn(epsilon)) * jnp.roll(jnp.eye(TRUE_NUM_STATES), 1, axis=1) \
                        + (0.05 + rn(epsilon)) / TRUE_NUM_STATES
    transition_matrix   = jnp.clip(transition_matrix, 0, None) # Ensures non-negativity
    transition_matrix   = normalize(transition_matrix)
    
    
    #TODO STUDY covariance and means matrix rules to generalize randomization properly for covariance and means
    emission_means = jnp.column_stack([
            0.1*jnp.cos(jnp.linspace(0, 2 * jnp.pi, TRUE_NUM_STATES + 1))[:-1],
            0.*jnp.sin(jnp.linspace(0, 2 * jnp.pi, TRUE_NUM_STATES + 1))[:-1],
            jnp.zeros((TRUE_NUM_STATES, EMISSION_DIM - 2))
        ])
    emission_covs   = jnp.tile(0.1**2 * jnp.eye(EMISSION_DIM), (TRUE_NUM_STATES, 1, 1))
    


    return initial_probs, transition_matrix, emission_means, emission_covs


'Initialize HMMs'
def init_teachers():
    initial_probs, transition_matrix, emission_means, emissions_cov = initial()
    
    T0, T0_props    = HMM.initialize(initial_probs=initial_probs,
                                    transition_matrix=transition_matrix,
                                    emission_means=emission_means,
                                    emission_covariances=emissions_cov)
    

    teacher_num = 1
    init, trans, means, covs = perturbation(teacher_num, epsilon, 
                                            initial_probs, 
                                            transition_matrix, 
                                            emission_means, 
                                            emissions_cov)
    T1, _           = HMM.initialize(initial_probs=init, transition_matrix=trans,
                                    emission_means=means, emission_covariances=covs)
    
    init, trans, means, covs = perturbation(2, epsilon, 
                                            init, 
                                            trans, 
                                            means, 
                                            covs)
    T2, _           = HMM.initialize(initial_probs=init, transition_matrix=trans,
                                    emission_means=means, emission_covariances=covs)
    
    'Baseline option 1'
    # n       = lambda data, new_data : multivariate_normal(mean=np.mean(data, axis=(0,1)), cov=np.cov(data.reshape(-1, EMISSION_DIM), rowvar=False)).pdf(new_data)
    # base    = lambda train, test : np.log(n(train, test)).sum(axis=1).mean(axis=0)
    'Baseline option 2'
    baseline_model = GaussianMixture(n_components=1)
    base = lambda train,test: baseline_model.fit(
            train.reshape(-1, EMISSION_DIM)
            ).score_samples(
                test.reshape(-1, EMISSION_DIM)
            ).reshape(NUM_TRIALS,NUM_TIMESTEPS).sum(axis=1).mean(axis=0)

    return T0, T1, T2, base




def generate_data_from_model(model, params, key, NUM_TRIALS, NUM_TIMESTEPS):
    """
    Sample many trials. 
    """
    keys = jr.split(key, NUM_TRIALS)
    sample_many_trials = vmap(model.sample, (None, 0, None), (0, 0))
    T0_states, emissions = sample_many_trials(
        params, keys, NUM_TIMESTEPS
    )
    return T0_states, emissions


'Generate datasets for training and testing'
def dgen(teachers):
    dataset = []

    gdata   = lambda params, key: generate_data_from_model(HMM, params, jr.PRNGKey(key), NUM_TRIALS, NUM_TIMESTEPS)
    for T in teachers:
        _, T_emissions_train        = gdata(T, 1)
        T_states, T_emissions_test  = gdata(T, 100)

        dataset.append([T, T_emissions_train, T_emissions_test])

    return dataset




'Train/fit the HMMs'
def train(teachers, S, S_props, hmm_student):
    fit = lambda hmm_class, params, props, emissions : zip(*[hmm_class.fit_em(param, prop, emissions) for param, prop in zip(*[params, props])])
    # single_fit = lambda hmm_class, params, props, emissions : hmm_class.fit_em(params, props, emissions, num_iters=NUM_EPOCHS)
    #TODO turn into a loop...
    S0, _   = fit(hmm_student, S, S_props, teachers[0][1])
    S1, _   = fit(hmm_student, S, S_props, teachers[1][1])
    S00, _  = fit(hmm_student, S0, S_props, teachers[0][1])
    S01, _  = fit(hmm_student, S0, S_props, teachers[1][1])
    S11, _  = fit(hmm_student, S1, S_props, teachers[1][1])
    # T01, _  = single_fit(hmm, T0, T0_props, T1_emissions_train)
    # S_l0, _ = fit(hmm_student, S_l, S_l_props, T0_emissions_train)

    return S0, S1, S00, S01, S11




'remove negative lines and nullify negative elements'
def rm_null(results):
    # Remove under-performing student
    removed = []
    keys_to_remove = []
    for key, value in results.items():
        if key == list(results.keys())[0]:
            continue  # Skip the header row
        
        if isinstance(value[0], list):  # Check if the value is a list of lists
            if all(max(sublist) < 0 for sublist in value):
                keys_to_remove.append(key)
        else:  # If it's a list of numbers
            if max(value) < 0:
                keys_to_remove.append(key)

    for key in keys_to_remove:
        del results[key]
        removed.append(key)

    # Remove under-performing seed-generated students sub-lists
    data = {}
    null_n = lambda list: [x if x >= 0 else 0 for x in list] #nullify negative
    for key, value in results.items():
        if key != list(results.keys())[0]: # list(results.keys())[0] = index title
            data[key] = [null_n(row) for row in value if any(x >= 0 for x in row)]
    
    return data, removed


'Convert dict to DF'
def df_conv(data, index, index_title):  
    result = {}
    for key, value in data.items():
        arr = np.array(value)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        result[key] = arr.T.tolist()
    
    df = pd.DataFrame(result, index=index)

    
    for col in df.columns:
        df[col] = df[col].apply(np.array)
    pd.set_option('display.max_colwidth', None)

    df.index.name = index_title # Bring the table title back

    #Switch columns, to bring
    df = df.reindex(sorted(df.columns, key=lambda x: x.split('_')[0]), axis=1) 

    df.to_csv('Params likelihood.csv')
    
    return df


def likelihood(students, teachers):
    evaluate_func = lambda hmm_class : vmap(hmm_class.marginal_log_prob, [None, 0], 0) #evaluate
    ev = lambda hmm, features, test: (evaluate_func(hmm)(features, test)).mean() #eval_true

    results = {"Likelihood over" : [f'T{i}' for i, _ in enumerate(teachers)]}
    keys = []
    for i in range(MAX_S_STATE - TRUE_NUM_STATES):
        keys.extend([f"S0_{i}", f"S1_{i}", f"S00_{i}", f"S01_{i}", f"S11_{i}"]) # ["T01" , T01, HMM] #TODO include teacher/s ]

    for key, models, hmm_type in zip(keys, [student[0] for student in students], [student[1] for student in students]):
        results[key] = []
        for model in models:
            results[key].append([float((ev(hmm_type, model, test)-base(train, test))/(ev(HMM, T, test)-base(train, test))) for T, train, test in teachers])
    
    data, removed_col = rm_null(results)

    index_title = list(results.keys())[0]
    index = results[index_title]
    
    df = df_conv(data, index, index_title)

    return df, removed_col




NUM_TRAIN_BATCHS  = 3
NUM_TEST_BATCHS    = 1

NUM_EPOCHS          = 3 #Increase the pochs before addin , num_iters=NUM_EPOCHS to fit_em
NUM_TIMESTEPS       = 100
NUM_TRIALS          = 1000
STUDENTS_NUM        = 2


'HMM Type and settings'
EMISSION_DIM    = 5
TRUE_NUM_STATES = 10
MAX_S_STATE     = TRUE_NUM_STATES + 20
epsilon         = 0.1
scale           = 0.1
HMM = GaussianHMM(TRUE_NUM_STATES, EMISSION_DIM)


if __name__ == '__main__':
    T0, T1, T2, base = init_teachers()
    teachers = [T0, T1, T2]
    teachers = dgen(teachers)

    students = []
    for num in range(TRUE_NUM_STATES, MAX_S_STATE):
        hmm_student = GaussianHMM(num, EMISSION_DIM)
        S, S_props = zip(*[hmm_student.initialize(jr.PRNGKey(key)) for key in range(STUDENTS_NUM)])

        students_data = train(teachers, S, S_props, hmm_student)
        students.extend([s, hmm_student] for s in students_data)


    df, removed_students = likelihood(students, teachers)

    print(df)
    print(f'\nRemoved models with low performance: {removed_students}')

    visualize(df)


    'Plot emissions and true_states in the emissions plane'
    # plot_gaussian_hmm(HMM, T0, T0_emissions_test[0], T0_states[0], title="True HMM emission distribution")

    'Plot emissions vs. time with background colored by true state'
    # plot_gaussian_hmm_data(HMM, T0, T0_emissions_test[0], T0_states[0])




#TODO add explanation to df about everything: "T0 = Ground truth, T1 = T0 + Perturbation, T2 = T1 + Perturbation... S = initial student, S0 = S Trained on T0, S1 = trained on T1, S01 = S0 trained on T1, Sijk = Sij trained on Tk, etc. "
'''
TODO
-report... 
1. In debug mode- how to run from a specific line while keeping the state?
2. Generalize student S_l
3. Keep function extractions and organizations for teachers, etc. 

...
4. ! Plan/compute algo for generalizing students (s_1..1,s_1..2...?) for 5 teachers (T5) - combinatorics computation:
1->x->y where y>=x>=1
5. Generalize teachers and students using the algorithms I developed and evaluate on an unseen teacher T6
6. Visualize again w/ various dim reductions methods (PCA), or just likelihood, over T6.
7. Try again for 99+1 teachers.
8. try for several emission dims, and other generalizations + check specifications (final level)
*optional: 1 graph per 1 teacher's emissions, have 5 graphs total, or merge them (using different colors per teachers emissions eval)
0. Check dynamax and Claude for how to pertub teachers properly (so the likelihood won't surpass the teacher? Or perhaps it is ok) VX - return to this step
'''

'''
What have I done:
1.  Perturbed teachers?
2.  Trained, evaluated, and compared students and teachers over teachers...
3.  Visualized initial true params emissions
    Evaluate all on unseen teacher T3 V
    Repeating each curriculum with many randomly initialized students. V
    Visualize results: Use rings on teacher's data, and graph the dataframe's data V
'''









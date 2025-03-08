from macros import *
import subprocess
import sys
import scipy.stats
import numpy as np
import pandas as pd
import optax
from add_extraneous import *
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
# from dynamax.hidden_markov_model import GaussianHMM
from dynamax.hidden_markov_model import DiagonalGaussianHMM
from dynamax.hidden_markov_model import SphericalGaussianHMM
from dynamax.hidden_markov_model import SharedCovarianceGaussianHMM

from visualize import *


'''
T0 = Ground truth
T1 = T0 + Perturbation
T2 = T1 + Perturbation
... 

S = initial student
S0 = S Trained on T0
S1 = trained on T1
S01 = S0 trained on T1
...
Sijk = Sij trained on Tk"
'''

def normalize(A):
    A /= A.sum(axis=1, keepdims=True) #Normalize rows
    # A = A.at[-1].set(1 - jnp.sum(A[:-1], axis=0)) #last row is defined
    # A = A.at[:, -1].set(1 - jnp.sum(A[:, :-1], axis=1)) #last column is defined
    return A


def perturbation(perturbation_num, epsilon, initial_probs, transition_matrix, emissions_means, emissions_cov):
    key = jr.PRNGKey(perturbation_num)
    np.random.seed(perturbation_num)

    initial_probs += epsilon*jr.uniform(key, shape=(TRUE_NUM_STATES,))
    initial_probs = initial_probs / jnp.sum(initial_probs)
    
    rn  = lambda x : jr.uniform(key, minval = -x, maxval = x)
    p   = rn(epsilon) * jnp.eye(TRUE_NUM_STATES) \
        + rn(epsilon) * jnp.roll(jnp.eye(TRUE_NUM_STATES), 1, axis=1) \
        + rn(epsilon) / TRUE_NUM_STATES
    transition_matrix   += p
    transition_matrix   = jnp.clip(transition_matrix, 0, None) # Ensures non-negativity
    transition_matrix   = normalize(transition_matrix)

    emissions_means += np.random.normal(-epsilon, epsilon, emissions_means.shape) 
    # emissions_cov   += np.random.normal(0, epsilon) * jnp.eye(EMISSION_DIM)[None, :, :]
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

    return T0, T1, T2



def create_students(teacher, students_num):
    students = []
    # Add perfect student
    #TODO atm not necessary. Just replace main code and run!

    # Add poor student
    for i in range(1, students_num+1):
        student = add_ring(teacher, ring_length=i)
        students.append(student)
    return students



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
def train(teachers, S, S_props, hmm_student):  #can also try with .fit_em
    fit = lambda hmm_class, params, props, emissions : zip(*[hmm_class.fit_sgd(param, prop, emissions, num_epochs=1, optimizer=optax.adam(1e-5)) for param, prop in zip(*[params, props])]) # for fit_em use max_iter
    teacher_fit = lambda hmm_class, params, props, emissions : hmm_class.fit_sgd(params, props, emissions) 


    loss        = []
    decodingST  = []
    decodingTS  = []
    for i in range(NUM_EPOCHS):
        print(f'checkpoint - train - iteraion: {i}')

        print(f'checkpoint - train - loss : {loss}')
        for model in S:
            decodingST.append([decoding(hmm_student, model, HMM, T, test) for T, train, test in teachers])
            print(f'checkpoint - train - decoding TS : {decodingST}')
            decodingTS.append([decoding(HMM, T, hmm_student, model, test) for T, train, test in teachers])
            print(f'checkpoint - train - decoding ST : {decodingTS}')

        S, l   = fit(hmm_student, S, S_props, teachers[0][1])
        loss = np.array(l) if len(loss) == 0 else np.concatenate((loss, l), axis=0)
        print(f'checkpoint - train - end of iteration')


    print(f'l = {loss},  {decodingST} ,  {decodingTS}  ')
    
    # S1, _   = fit(hmm_student, S, S_props, teachers[1][1])
    # S00, _  = fit(hmm_student, S0, S_props, teachers[0][1])
    # S01, _  = fit(hmm_student, S0, S_props, teachers[1][1])
    # S11, _  = fit(hmm_student, S1, S_props, teachers[1][1])
    # T01, _  = single_fit(hmm, T0, T0_props, T1_emissio ns_train)
    # S_l0, _ = fit(hmm_student, S_l, S_l_props, T0_emissions_train)

    # return S0, S1, S00, S01, S11
    return loss, decodingST, decodingTS



'remove negative lines and nullify negative elements'
def rm_null(results):
    # Remove under-performing student
    removed = []
    keys_to_remove = []
    for key, value in results.items():
        if key == list(results.keys())[0]:
            continue  # Skip the header row
        
        if isinstance(value[0], list):  # Check if the value is a list of lists #TODO no need to check..
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
            # data[key] = [null_n(row) for row in value if any(x >= 0 for x in row)]  TODO ?
            data[key] = [row for row in value] 
    
    return data, removed
    # return results, removed



'Convert dict to DF'
def df_conv(results):  
    result = {}
    for i, data in enumerate(results):
        index_title = list(data.keys())[0]
        index = data[index_title]
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
        # df = df.reindex(sorted(df.columns, key=lambda x: x.split('_')[0]), axis=1)  #TODO Remove if not working or needed

        df.to_csv(f'Params likelihood S->T.csv') if i == 0 else df.to_csv(f'Params likelihood T->S.csv')
    
    return df



def likelihood(students, teachers):
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
    

    evaluate_func = lambda hmm_class : vmap(hmm_class.marginal_log_prob, [None, 0], 0) #evaluate
    ev = lambda hmm, features, test: (evaluate_func(hmm)(features, test)).mean() #eval_true

    results = {"Likelihood over" : [f'T{i}' for i, _ in enumerate(teachers)]}
    keys = []

    #We duplicate the keys in the eeded amount to include students with a larger number of states
    for _ in range(MAX_S_STATE - TRUE_NUM_STATES):
        keys.extend(f"S{key}" for key in S_KEYS) # ["T01" , T01, HMM] #TODO include teacher/s ]

    # results.update({key: [] for key in S_KEYS}) TODO doesn't work
    for k in S_KEYS:
        results[f'S{k}'] = []


    # adding student likelihoods to the DF columns (Sk with min num of states, Sk with min num of states+1, ...)
    for key, models, hmm_type in zip(keys, [student[0] for student in students], [student[1] for student in students]):
        #will loop based on number of students (student_num)
        for model in models:
            results[key].append([float((ev(hmm_type, model, test)-base(train, test))/(ev(HMM, T, test)-base(train, test))) for T, train, test in teachers])

    # adding teachers likelihoods to the DF columns
    for i, teacher in enumerate(teachers):
        results[f'T{i}'] = [float((ev(HMM, teacher[0], test)-base(train, test))/(ev(HMM, T, test)-base(train, test))) for T, train, test in teachers]

    
    data, removed_col = rm_null(results)

    index_title = list(results.keys())[0]
    index = results[index_title]
    
    df = df_conv(data) #, index, index_title)

    return df, removed_col



def decode(students, teachers):
    resultsST = {"S->T Decoding" : [f'T{i}' for i, _ in enumerate(teachers)]}
    resultsTS = {"T->S Decoding" : [f'T{i}' for i, _ in enumerate(teachers)]}
    keys = []

    #We duplicate the keys in the eeded amount to include students with a larger number of states
    for _ in range(MAX_S_STATE - TRUE_NUM_STATES):
        keys.extend(f"S{key}" for key in S_KEYS) # ["T01" , T01, HMM] #TODO include teacher/s 

    # results.update({key: [] for key in S_KEYS}) TODO doesn't work
    for k in S_KEYS:
        resultsST[f'S{k}'] = []
        resultsTS[f'S{k}'] = []


    # adding student likelihoods to the DF columns (Sk with min num of states, Sk with min num of states+1, ...)
    for key, student, hmm_type in zip(keys, [s for s in students], [student[1] for student in students]):
        for model in student[0]: #will loop based on number of students (student_num)
            resultsST[key].append([decoding(student[1], model, HMM, T, test) for T, train, test in teachers])
            resultsTS[key].append([decoding(HMM, T, student[1], model, test) for T, train, test in teachers])


    # adding teachers likelihoods to the DF columns
    for i, teacher in enumerate(teachers):
        resultsST[f'T{i}'] = [decoding(HMM, teacher[0], HMM, T, test) for T, train, test in teachers]
        resultsTS[f'T{i}'] = [decoding(HMM, T, HMM, teacher[0], test) for T, train, test in teachers]

    
    # data, removed_col = rm_null(results)

    results = [resultsST, resultsTS]
    
    df = df_conv(results)

    return df


if __name__ == '__main__':
    print('checkpoint 1')

    # # if True:
    #     # results = load_csv()

    # # else: 
    # T0, T1, T2 = init_teachers()
    # teachers = [T0, T1, T2]
    # teachers = dgen(teachers)

    # print('checkpoint 2')
    # # students shape : [S0_minStates, S1_minStates, ... Sk_minStates, S0_minStates+1, S1_minStates+1, ...]
    # students = []
    # for num in range(MIN_S_STATE, MAX_S_STATE):
    #     hmm_student = GaussianHMM(num, EMISSION_DIM)
    #     S, S_props = zip(*[hmm_student.initialize(jr.PRNGKey(key)) for key in range(STUDENTS_NUM)])
    #     # students_data = train(teachers, S, S_props, hmm_student)
    #     # students.extend([s, hmm_student] for s in students_data)
    #     print('checkpoint 3')
    #     loss, decodingST, decodingTS = train(teachers, S, S_props, hmm_student)

    # print('checkpoint 4')

    # plot_decodingEpochs(loss, decodingST, decodingTS, num_epochs=NUM_EPOCHS)


    plot_decodingEpochs(None, None, None, num_epochs=NUM_EPOCHS)




    # performance_plot_3D(df)
    # performance_plot(results)

    # transitions_plot(students[0][1], ) for later... 

    'Plot emissions and true_states in the emissions plane'
    # plot_gaussian_hmm(HMM, T0, T0_emissions_test[0], T0_states[0], title="True HMM emission distribution")

    'Plot emissions vs. time with background colored by true state'
    # plot_gaussian_hmm_data(HMM, T0, T0_emissions_test[0], T0_states[0])



#TODO Long term
'''
0. Deep thinking and reasoning (w/ gpt?) about what’s next
0.1 Zoom in on the first epoch - reduce number of iteratios (see message from KABI)
1. Decode with likelihood- how are the students (different seed) changes over the epochs during fit?
2. Then train on T1, T2... and train S0 again on T1
3. try changing perturbation for an optimization one (like SGD), and add more T0 teachers (different seeds)
4. add cross decoding ?

Goal - find a model that does well on T1, should also do reasonably well on T0 or vice versa?


TODO - Specific tasks, keep an open mind (Can be CHANGED!)
1. Imporove style of graphs, plan ahead to make it easier before second run with data/new students
2. Add students like before
3. Add some teachers? perhaps with different perturbations
Go to step 5 - to test for each epoch, rewrtie train(..) function : hmm_class.fit_em(params, prop, emissions, num_iters=1) , i.e. NUM_EPOCHS=1, and add a loop, and test each epoch ->
Create a good student - duplciate teacher, and add some states I cannot rach (sanity check that it works)


TODO - side taskquests
- Update add_extranaous ring func to fit the new HMM. Use GPT

1. In debug mode- how to run from a specific line while keeping the state? 
2. Hot decode/reload ? Ask Community
3. Finish all todo's quests



Alternative tasks (archive):
. ! Plan/compute algo for generalizing students (s_1..1,s_1..2...?) for 5 teachers (T5) - combinatorics computation:
1->x->y where y>=x>=1
. Generalize teachers and students using the algorithms I developed and evaluate on an unseen teacher T6
. Visualize again w/ various dim reductions methods (PCA), or just likelihood, over T6.
. Try again for 99+1 teachers.
. try for several emission dims, and other generalizations + check specifications (final level)
*optional: 1 graph per 1 teacher's emissions, have 5 graphs total, or merge them (using different colors per teachers emissions eval)
. Check dynamax and Claude for how to pertub teachers properly (so the likelihood won't surpass the teacher? Or perhaps it is ok) VX - return to this step



'''


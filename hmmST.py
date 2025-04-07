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


# def pert(model, id):
#     key = jr.PRNGKey(id)
#     np.random.seed(id)

#     initial_probs += epsilon*jr.uniform(key, shape=(TRUE_NUM_STATES,))
#     initial_probs = initial_probs / jnp.sum(initial_probs)
    
#     rn  = lambda x : jr.uniform(key, minval = -x, maxval = x)
#     p   = rn(epsilon) * jnp.eye(TRUE_NUM_STATES) \
#         + rn(epsilon) * jnp.roll(jnp.eye(TRUE_NUM_STATES), 1, axis=1) \
#         + rn(epsilon) / TRUE_NUM_STATES
#     transition_matrix   += p
#     transition_matrix   = jnp.clip(transition_matrix, 0, None) # Ensures non-negativity
#     transition_matrix   = normalize(transition_matrix)

#     emissions_means += np.random.normal(-epsilon, epsilon, emissions_means.shape) 
#     # emissions_cov   += np.random.normal(0, epsilon) * jnp.eye(EMISSION_DIM)[None, :, :]
#     emissions_cov   += np.random.normal(-epsilon, epsilon) * jnp.eye(EMISSION_DIM)[None, :, :]
#     emissions_cov   = (emissions_cov + np.swapaxes(emissions_cov, -2, -1)) / 2 #Create PSD part 1 : ( A + A.T )/ 2
#     # Create PSD part 2 : Make diagonal absolute value:
#     abs_diags = jnp.abs(jnp.diagonal(emissions_cov, axis1=-2, axis2=-1))
#     mask = jnp.eye(emissions_cov.shape[-1]).astype(bool) #Mask 
#     emissions_cov = jnp.where(mask, abs_diags[..., jnp.newaxis], emissions_cov) 

#     # print(emissions_cov)
#     return initial_probs, transition_matrix, emissions_means, emissions_cov



'Create HMM parameters for Ts'
def initial(): 
    'Specify initial parameters of the HMM' # TODO change to if Model is not set. if set - transition matrix = model.transition, etc.
    key = jr.PRNGKey(1)  # Random seed
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
def init_teachers(initial_probs, transition_matrix, emission_means, emissions_cov):    
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
    T1, T1_props            = HMM.initialize(initial_probs=init, transition_matrix=trans,
                                    emission_means=means, emission_covariances=covs)
    
    init, trans, means, covs = perturbation(2, epsilon, 
                                            init, 
                                            trans, 
                                            means, 
                                            covs)
    T2, T2_props            = HMM.initialize(initial_probs=init, transition_matrix=trans,
                                    emission_means=means, emission_covariances=covs)

    return [T0, T1, T2], [T0_props, T1_props, T2_props]


'This will create students with 100% accuracy on liklihood (emissions), but different structure and hence, high decoding (error value)'
def create_students(initial_probs, transition_matrix, emission_means, emissions_cov, ring=True):
    students = []
    
    for n in range(1, STUDENTS_NUM+1): # n = number of rings
        if ring:
            student_params, props, hmm_type = add_ring(initial_probs, transition_matrix, emission_means, emissions_cov, ring_length=n)
            students.append([student_params, props, hmm_type])
        hmm_type = GaussianHMM(MAX_S_STATE, EMISSION_DIM) #TODO: Generalize for students w/ different num of states
        student_params, props = hmm_type.initialize(jr.PRNGKey(n))
        if DEBUG and VERBOSE:
            print(f'random params = {student_params}') 
        students.append([student_params, props, hmm_type])
    return students



def generate_data_from_model(model, params, key, NUM_TRIALS, NUM_TIMESTEPS):
    """
    Sample many trials. 
    """
    keys = jr.split(key, NUM_TRIALS)
    sample_many_trials = vmap(model.sample, (None, 0, None), (0, 0))
    states, emissions = sample_many_trials(
        params, keys, NUM_TIMESTEPS
    )
    return states, emissions



'Generate datasets for training and testing'
def dgen(teachers):
    dataset = []

    gdata_em    = lambda params, key: generate_data_from_model(HMM, params, jr.PRNGKey(key), NUM_TRIALS, NUM_TIMESTEPS)
    gdata_sgd   = lambda params, key : vmap(partial(HMM.sample, params, num_timesteps=NUM_TIMESTEPS))(
        jr.split(jr.PRNGKey(key), NUM_TRAIN_BATCHS))
    gdata       = lambda params, key : (gdata_sgd if SGD else gdata_em)(params, key)


    states = []
    for T in teachers:
        _, T_emissions_train        = gdata(T, 42)
        T_states, T_emissions_test  = gdata(T, 99)

        dataset.append([T, T_emissions_train, T_emissions_test])
        states.append(T_states)
    
    return dataset, states



def fit_students(students, emissions):
    fit_em  = lambda hmm_class, params, props, emissions : hmm_class.fit_em(params, props, emissions, num_iters=ITER)
    fit_sgd = lambda hmm_class, params, props, opt_states, emissions : hmm_class.fit_sgd(params, props, emissions, init_opt_state=opt_states, num_epochs=ITER, optimizer=optax.adam(LEARNING_RATE))
    # fit     = lambda hmm_class, params, props, emissions : (fit_sgd if SGD else fit_em)(hmm_class, params, props, emissions) TODO clean

    trained_students = []
    for student in students:
        opt_states = None
        if len(student) == 3: #first run, without opt_states
            student_params, props, hmm_type = student
        else: 
            student_params, props, hmm_type, opt_states = student
        if SGD:
            student_params, opt_states, losses  = fit_sgd(hmm_type,  student_params, props, opt_states, emissions)
        else:
            student_params, losses  = fit_em(hmm_type,  student_params, props, emissions)
        trained_students.append([student_params, props, hmm_type, opt_states])
    return trained_students


'Train/fit the HMMs'
def train(teachers, students, teacher_focus_i): 
    # For fit_em use num_iters # optimizer=optax.adam(LEARNING_RATE). Look at ssm.py to find fit_sgd implementation
    teacher_fit = lambda hmm_class, params, props, emissions : hmm_class.fit_sgd(params, props, emissions) 

    test_likelihoods = []
    train_likelihoods = []
    decodingST  = [] 
    decodingTS  = []
    for i in range(NUM_EPOCHS):
        print(f'Train - iteraion: {i}')
        epoch_decodingST = []
        epoch_decodingTS = []
        epoch_test_likelihoods = []
        epoch_train_likelihoods = []

        for student in students:
            hmm_student = student[2]
            params_student = student[0]
            # epoch_decodingST.append([decoding(hmm_student, params_student, HMM, T, test) for T, train, test in teachers])
            epoch_decodingST.append([decoding(hmm_student, params_student, HMM, T, test) for T, train, test in teachers]) 
            # epoch_decodingST.append([0.0 for T, train, test in teachers])
            epoch_decodingTS.append([decoding(HMM, T, hmm_student, params_student, test) for T, train, test in teachers])
            # epoch_decodingTS.append([0.0 for T, train, test in teachers])
            epoch_test_likelihoods.append([max(likelihood(hmm_student, params_student, T, train, test), 0) for T, train, test in teachers])
            epoch_train_likelihoods.append([max(likelihood(hmm_student, params_student, T, train, train), 0) for T, train, test in teachers])
        

        decodingTS.append(epoch_decodingST)
        decodingST.append(epoch_decodingTS)
        test_likelihoods.append(epoch_test_likelihoods)
        train_likelihoods.append(epoch_train_likelihoods)

        if teacher_focus_i == -1: #meaning multi-teacher fitting is selected
            students = fit_students(students, teachers[i%3][1])
        else:
            students = fit_students(students, teachers[teacher_focus_i][1])


    '''
    The format of decoding and liklihoods:
    epoch        S0                               S1
    1       [[T0, T1, T2] [T0, T1, T2] ... ]
    2       ...
    ..

    In the csv each element is a 2D matrix, the rows represent 
    '''
    return train_likelihoods, test_likelihoods, decodingST, decodingTS, students #[s[0:3] for s in students] #TODO - Need to return new states too for better training.




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



def likelihood(student_type, student_params, teacher, teacher_train_obs, test_obs):
    baseline_model = GaussianMixture(n_components=1)
    base = lambda train, test: baseline_model.fit(
        train.reshape(-1, EMISSION_DIM)
    ).score_samples(
        test.reshape(-1, EMISSION_DIM)
    ).reshape(-1, NUM_TIMESTEPS).sum(axis=1).mean(axis=0)
    evaluate_func = lambda hmm_class: vmap(hmm_class.marginal_log_prob, [None, 0], 0)  # evaluate
    ev = lambda hmm, features, test: (evaluate_func(hmm)(features, test)).mean()  # eval_true

    if VERBOSE:
        def evaluate_model(hmm, model, test_data, base=False):
            #     if base:
            #         true_loss = vmap(partial(hmm.marginal_log_prob, model))(test_data).sum()
            #         # true_loss = vmap(partial(hmm._estimate_log_prob  else hmm., model))(test_data).sum()
            #     true_loss += hmm.log_prior(model)
            #     true_loss = -true_loss / test_data.size
            #     return true_loss
            # base = lambda train,test: baseline_model.fit(
            #     train.reshape(-1, EMISSION_DIM)
            #     ).score_samples(
            #         test.reshape(-1, EMISSION_DIM)
            #     )
            pass

        print(f'Base liklihood: {base(teacher_train_obs, test_obs)}')
        print(f'Teacher liklihood: {ev(HMM, teacher, test_obs)}')
        print(f'Student liklihood: {ev(student_type, student_params, test_obs)}')
        print(f'Student train liklihood: {ev(student_type, student_params, teacher_train_obs)} \n\n')

        # print(f'Teacher new eval: {evaluate_model(HMM, teacher, test_obs)} \n\n')
        # print(f'Student new eval: {evaluate_model(student_type, student_params, test_obs)} \n\n')
        # print(f'Student new eval train: {evaluate_model(student_type, student_params, teacher_train_obs)} \n\n')
        # print(f'Base new eval: {evaluate_model(baseline_model, base(teacher_train_obs, test_obs), test_obs, base=True)} \n\n')

    student_score = ev(student_type, student_params, test_obs)
    base_score = base(teacher_train_obs, test_obs)
    teacher_score = ev(HMM, teacher, test_obs)
    return float((student_score - base_score) / (teacher_score - base_score))



def likelihoods(students, teachers):
    'Baseline option 1'
    # n       = lambda data, new_data : multivariate_normal(mean=np.mean(data, axis=(0,1)), cov=np.cov(data.reshape(-1, EMISSION_DIM), rowvar=False)).pdf(new_data)
    # base    = lambda train, test : np.log(n(train, test)).sum(axis=1).mean(axis=0)
    'Baseline option 2'
    baseline_model = GaussianMixture(n_components=1)
    base = lambda train,test: baseline_model.fit(
            train.reshape(-1, EMISSION_DIM)
            ).score_samples(
                test.reshape(-1, EMISSION_DIM)
            ).reshape(NUM_TRIALS, NUM_TIMESTEPS).sum(axis=1).mean(axis=0)
    

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
    -----------------------------

    students shape : 
    [S0_minStates, S1_minStates, ... Sk_minStates, S0_minStates+1, S1_minStates+1, ...]

    The format of decoding and liklihoods:
    epoch        S0                               S1
    1       [[T0, T1, T2] [T0, T1, T2] ... ]
    2       ...
    ..
    *In the csv each element is a 2D matrix
    '''
    if false or DEBUG: #False = plot graphs, True = plot and train
        print('checkpoint 1')
        initial_probs, transition_matrix, emission_means, emissions_cov = initial()
        
        # HMM params = params.initial, params.transitions,  params.emissions
        [T0, T1, T2], [T0_props, T1_props, T2_props] = init_teachers(initial_probs, transition_matrix, emission_means, emissions_cov)
        teachers = [T0, T1, T2]
        teachers_copy = [[deepcopy(T0), T0_props, HMM], [deepcopy(T1), T1_props, HMM], [deepcopy(T2), T2_props, HMM]]
        
        teachers, states = dgen(teachers)

        students = create_students(initial_probs, transition_matrix, emission_means, emissions_cov, ring = RING)

        if DEBUG:
            print('state = ', states[0].shape)
            print('emissions = ', teachers[0][2][0].shape)

            train_key, val_key, test_key = jr.split(jr.PRNGKey(0), 3)
            f = vmap(partial(HMM.sample, T0, num_timesteps=NUM_TIMESTEPS))
            train_true_states, train_emissions = f(jr.split(train_key, NUM_TRAIN_BATCHS))
            test_true_states,  test_emissions  = f(jr.split(test_key, NUM_TEST_BATCHS))
            
            for i in range(NUM_EPOCHS):
                likelihood(students[0][2], students[0][0], teachers[0][0], train_emissions, test_true_states)
                students = fit_students(students, teachers[0][1])

            
            # plot_gaussian_hmm(HMM, T0, train_emissions[0], train_true_states[0], 
            #                 title="True HMM emission distribution")
            # plot_gaussian_hmm(HMM, T0, teachers[0][2][0], states[0][0])

        else:
            #First training on T0
            print('checkpoint 2')
            if MULTI_TEACHERS_FIT:
                s_train_likelihoods, s_test_likelihoods, s_decodingST, s_decodingTS, trained_students = train(teachers, students, -1) #-1 = multi teachers training
                t_train_likelihoods, t_test_likelihoods, t_decodingST, t_decodingTS, trained_teachers = train(teachers, teachers_copy, -1) 
            else:            
                s_train_likelihoods, s_test_likelihoods, s_decodingST, s_decodingTS, trained_students = train(teachers, students, 0)
                t_train_likelihoods, t_test_likelihoods, t_decodingST, t_decodingTS, trained_teachers = train(teachers, teachers_copy, 0) 
            

            run_params = f'{NUM_EPOCHS}Epochs_{ITER}Iter_{NUM_TIMESTEPS}Timesteps_{NUM_TRIALS}Trials_SGD={SGD}_DEBUG={DEBUG}'
            # filename = f'multi-T-fit_{run_params}'
            filename = f'multi-T-fit_{run_params}' if MULTI_TEACHERS_FIT else f'T0-fit_{run_params}'
            plot_decodingEpochs(s_train_likelihoods, s_test_likelihoods, s_decodingST, s_decodingTS, \
                                    t_train_likelihoods, t_test_likelihoods, t_decodingST, t_decodingTS,
                                    csv_file_param = f'{filename}')
            # plot_decodingEpochs_singleModelType(train_likelihoods, test_likelihoods, decodingST, decodingTS, csv_file_name=f't_multi-T-fit_{run_params}')

            if not MULTI_TEACHERS_FIT:
                for teacher_focus_i in range(1, len(teachers)): #Choose on which teacher we're training next
                    s_train_likelihoods, s_test_likelihoods, s_decodingST, s_decodingTS, _ = train(teachers, trained_students, teacher_focus_i) 
                    t_train_likelihoods, t_test_likelihoods, t_decodingST, t_decodingTS, _ = train(teachers, trained_teachers, teacher_focus_i) 

                    filename = f'T{teacher_focus_i}-fit_{run_params}'
                    plot_decodingEpochs(s_train_likelihoods, s_test_likelihoods, s_decodingST, s_decodingTS, \
                                        t_train_likelihoods, t_test_likelihoods, t_decodingST, t_decodingTS,
                                        csv_file_param=f'{filename}')

    else:
        file_params = ""
        plot_decodingEpochs_singleModelType(csv_file_name=f's_{file_params}')
        plot_decodingEpochs_singleModelType(csv_file_name=f't_{file_params}')

        plot_decodingEpochs(csv_file_param=f'{file_params}')


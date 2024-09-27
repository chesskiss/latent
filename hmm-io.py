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

    initial_probs += jr.uniform(key, shape=(true_num_states,))
    initial_probs = initial_probs / jnp.sum(initial_probs)
    
    rn  = lambda x : jr.uniform(key, minval = -x, maxval = x)
    p   = rn(epsilon) * jnp.eye(true_num_states) \
        + rn(epsilon) * jnp.roll(jnp.eye(true_num_states), 1, axis=1) \
        + rn(epsilon) / true_num_states
    transition_matrix   += p
    transition_matrix   = jnp.clip(transition_matrix, 0, None) # Ensures non-negativity
    transition_matrix   = normalize(transition_matrix)

    emissions_means += np.random.normal(-epsilon, epsilon, emission_means.shape) 
    emissions_cov   += np.random.normal(0, epsilon) * jnp.eye(EMISSION_DIM)[None, :, :]
    emissions_cov   += np.random.normal(-epsilon, epsilon) * jnp.eye(EMISSION_DIM)[None, :, :]
    emissions_cov   = (emissions_cov + np.swapaxes(emissions_cov, -2, -1)) / 2 #Create PSD part 1 : ( A + A.T )/ 2
    # Create PSD part 2 : Make diagonal absolute value:
    abs_diags = jnp.abs(jnp.diagonal(emissions_cov, axis1=-2, axis2=-1))
    mask = jnp.eye(emissions_cov.shape[-1]).astype(bool) #Mask 
    emissions_cov = jnp.where(mask, abs_diags[..., jnp.newaxis], emissions_cov) 

    # print(emissions_cov)
    return initial_probs, transition_matrix, emission_means, emissions_cov

    

def initial(): 
    'Specify initial parameters of the HMM' # TODO change to if Model is not set. if set - transition matrix = model.transition, etc.
    key = jr.PRNGKey(0)  # Random seed
    initial_probs = jr.uniform(key, shape=(true_num_states,),  minval=0)
    initial_probs = initial_probs / jnp.sum(initial_probs)

    # initial_probs = jnp.ones(true_num_states) / true_num_states

    rn = lambda x : jr.uniform(key, minval = -x, maxval = x)
    transition_matrix   = (0.80 + rn(epsilon)) * jnp.eye(true_num_states) \
                        + (0.15 + rn(epsilon)) * jnp.roll(jnp.eye(true_num_states), 1, axis=1) \
                        + (0.05 + rn(epsilon)) / true_num_states
    transition_matrix   = jnp.clip(transition_matrix, 0, None) # Ensures non-negativity
    transition_matrix   = normalize(transition_matrix)
    
    
    #TODO STUDY covariance and means matrix rules to generalize randomization properly for covariance and means
    emission_means = jnp.column_stack([
            0.1*jnp.cos(jnp.linspace(0, 2 * jnp.pi, true_num_states + 1))[:-1],
            0.*jnp.sin(jnp.linspace(0, 2 * jnp.pi, true_num_states + 1))[:-1],
            jnp.zeros((true_num_states, EMISSION_DIM - 2))
        ])
    emission_covs   = jnp.tile(0.1**2 * jnp.eye(EMISSION_DIM), (true_num_states, 1, 1))
    


    return initial_probs, transition_matrix, emission_means, emission_covs



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



NUM_TRAIN_BATCHS  = 3
NUM_TEST_BATCHS    = 1

NUM_EPOCHS          = 100000
NUM_TIMESTEPS       = 100
NUM_TRIALS          = 1000
STUDENTS_NUM        = 100
epsilon             = 0.1
scale               = 0.1

'HMM Type and settings'
true_num_states = 10
EMISSION_DIM = 2
hmm = GaussianHMM(true_num_states, EMISSION_DIM)
hmm_n = GaussianHMM(2, EMISSION_DIM)



if __name__ == '__main__':
    'Initialize HMMs'
    initial_probs, transition_matrix, emission_means, emissions_cov = initial()
    
    T0, T0_props    = hmm.initialize(initial_probs=initial_probs,
                                    transition_matrix=transition_matrix,
                                    emission_means=emission_means,
                                    emission_covariances=emissions_cov)
    

    teacher_num = 1
    init, trans, means, covs = perturbation(teacher_num, epsilon, 
                                            initial_probs, 
                                            transition_matrix, 
                                            emission_means, 
                                            emissions_cov)
    T1, _           = hmm.initialize(initial_probs=init, transition_matrix=trans,
                                    emission_means=means, emission_covariances=covs)
    
    init, trans, means, covs = perturbation(2, epsilon, 
                                            init, 
                                            trans, 
                                            means, 
                                            covs)
    T2, _           = hmm.initialize(initial_probs=init, transition_matrix=trans,
                                    emission_means=means, emission_covariances=covs)
    

    # S, S_props = hmm.initialize(jr.PRNGKey(1000)) 
    # students_init   = [hmm.initialize(jr.PRNGKey(key)) for key in range(STUDENTS_NUM)]
    S, S_props = zip(*[hmm.initialize(jr.PRNGKey(key)) for key in range(STUDENTS_NUM)])

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

    S_l, S_l_props= hmm_n.initialize(jr.PRNGKey(1000))


    'Generate datasets'
    gdata   = lambda params, key: generate_data_from_model(hmm, params, jr.PRNGKey(key), NUM_TRIALS, NUM_TIMESTEPS)
    _, T0_emissions_train           = gdata(T0, 1)
    T0_states, T0_emissions_test    = gdata(T0, 100)
    _, T1_emissions_train           = gdata(T1, 1)
    T1_states, T1_emissions_test    = gdata(T1, 100)
    _, T2_emissions_train           = gdata(T2, 1)
    T2_states, T2_emissions_test    = gdata(T2, 100)
    teachers = [[T0, T0_emissions_train, T0_emissions_test], [T1, T1_emissions_train, T1_emissions_test], [T2, T2_emissions_train, T2_emissions_test]]


    'Plot emissions and true_states in the emissions plane'
    # plot_gaussian_hmm(hmm, T0, T0_emissions_test[0], T0_states[0], title="True HMM emission distribution")

    'Plot emissions vs. time with background colored by true state'
    # plot_gaussian_hmm_data(hmm, T0, T0_emissions_test[0], T0_states[0])



    'Train'
    fit  = lambda hmm_class, params, props, emissions : hmm_class.fit_em(params, props, emissions) #TODO add NUM_EPOCHS=NUM_EPOCHS
    S0, _       = fit(hmm, S, S_props, T0_emissions_train)
    S1, _       = fit(hmm, S, S_props, T1_emissions_train)
    S00, _      = fit(hmm, S0, S_props, T0_emissions_train)
    S01, _      = fit(hmm, S0, S_props, T1_emissions_train)
    S11, _      = fit(hmm, S1, S_props, T1_emissions_train)
    T01, _      = fit(hmm, T0, T0_props, T1_emissions_train)
    S_l0, _     = fit(hmm_n, S_l, S_l_props, T0_emissions_train)

    evaluate_func = lambda hmm_class : vmap(hmm_class.marginal_log_prob, [None, 0], 0) #evaluate
    ev = lambda hmm, features, test: (evaluate_func(hmm)(features, test)).mean() #eval_true

    params = [
        ["T0" , T0 , hmm],
        ["T1" , T1 , hmm],
        ["T2" , T2 , hmm],
        ["S"  , S  , hmm],
        ["S0" , S0 , hmm],
        ["S1" , S1 , hmm],
        ["S00", S00, hmm],
        ["S01", S01, hmm],
        ["S11", S11, hmm],
        ["T01" , T01, hmm],
        ["S_l", S_l, hmm_n],
        ["S_l0",S_l0, hmm_n]
    ]


    #TODO add explanation to df about everything: "T0 = Ground truth, T1 = T0 + Perturbation, T2 = T1 + Perturbation... S = initial student, S0 = S Trained on T0, S1 = trained on T1, S01 = S0 trained on T1, Sijk = Sij trained on Tk, etc. "
    results = {"Likelihood over" : ["T0", "T1", "T2"]}

    results_unormalized = {"Likelihood over" : ["T0", "T1", "T2"]}


    # for key, model, hmm_type in params:
    #         for T, train, test in teachers:
    #             print((ev(hmm_type, model, test)-base(train, test))/(ev(hmm, T, test)-base(train, test)))
            # print(f'base  = {base(train, test)}')
            # print(f' init = {ev(hmm,S0,test)}')
            # print(f' S0 = {ev(hmm, S0, test)}')
            # print(f' T = {ev(hmm, T, test)}')
            # print((ev_l(S_l1, test, hmm_n)-base(train, test))/(ev(T, test)-base(train, test)))
    # results['base'] = [base(train, test)  for T, train, test in teachers]



    removed = []
    for key, model, hmm_type in params:
        results[key] = [float((ev(hmm_type, model, test)-base(train, test))/(ev(hmm, T, test)-base(train, test))) for T, train, test in teachers] 
        # results_unormalized[key] = [ev(hmm_type, model, test) for _, _, test in teachers] 
        if max(results[key])<0:
            del results[key]
            removed.append(key)
    
    df1 = pd.DataFrame(results)
    # df2 = pd.DataFrame(results_unormalized)

    # df = pd.concat([df1, df2])
    df=df1

    df.to_csv('Params likelihood.csv')

    print(f'\nRemoved models with low performance: {removed}')

    df = nullify_negative(df)
    visualize(df)



    

    # for i in range(5):
    #     student_states_trained_perturbed = hmm.most_likely_states(peterbuted_student_params_trained, T0_emissions_test[i])
    #     student_states_trained = hmm.most_likely_states(student_params_trained, T0_emissions_test[i])
    #     student_states_initial = hmm.most_likely_states(student_params_initial, T0_emissions_test[i])
    #     print('trial = ', i)
    #     print('student initial ', student_states_initial)
    #     print('student trained ', student_states_trained)
    #     print('student perturbed trained ', student_states_trained_perturbed)
    #     print('teacher ', T0_states[i])
    # params = [T0, student_params_initial, student_params_trained]


    # plot_m_arr(params, EMISSION_DIM, true_num_states)

'''
TODO
0. Check dynamax and Claude for how to pertub teachers properly (so the likelihood won't surpass the teacher? Or perhaps it is ok) VX - return to this step
1. Evaluate all on unseen teacher T3 V
2. Repeating each curriculum with many randomly initialized students.
3. Visualize results: Use rings on teacher's data, and graph the dataframe's data 
4. ! Plan/compute algo for generalizing students (s_1..1,s_1..2...?) for 5 teachers (T5) - combinatorics computation:
1->x->y where y>=x>=1
5. Generalize teachers and students using the algorithms I developed and evaluate on an unseen teacher T6
6. Visualize again w/ various dim reductions methods (PCA), or just likelihood, over T6.
7. Try again for 99+1 teachers.
8. try for several emission dims, and other generalizations + check specifications (final level)
*optional: 1 graph per 1 teacher's emissions, have 5 graphs total, or merge them (using different colors per teachers emissions eval)
'''

'''
What have I done:
1. Perturbed teachers?
2. Trained, evaluated, and compared students and teachers over teachers...
3. Visualized initial true params emissions
'''









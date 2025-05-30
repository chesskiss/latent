from macros import *
import subprocess
import sys
import numpy as np
from add_extraneous import *
import dynamax

from functools import partial

from jax import vmap
import jax.numpy as jnp
import jax.random as jr
from scipy.stats import multivariate_normal

from sklearn.mixture import GaussianMixture 
from visualize import *


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



STUDENTS_NUM        = 5




import jax.numpy as jnp
import jax

def backup_hmm(params, props, filename="hmm_backup_jax.pkl"):
    jax.tree_util.tree_map(lambda x: jnp.save(filename, x), (params, props))
    print("Backup completed.")

def restore_hmm(filename="hmm_backup_jax.pkl"):
    params, props = jax.tree_util.tree_map(lambda x: jnp.load(filename, allow_pickle=True), (None, None))
    print("Restoration completed.")
    return params, props


if __name__ == '__main__':
    'Initialize HMMs'
    initial_probs, transition_matrix, emission_means, emissions_cov = initial()
    
    T0, T0_props    = HMM.initialize(initial_probs=initial_probs,
                                    transition_matrix=transition_matrix,
                                    emission_means=emission_means,
                                    emission_covariances=emissions_cov)


    student_hmm = GaussianHMM(MAX_S_STATE, EMISSION_DIM)
    T, T0_props = student_hmm.initialize(jr.PRNGKey(10))
    S, S_props = student_hmm.initialize(jr.PRNGKey(0))
    # S, S_props = zip(*[student_hmm.initialize(jr.PRNGKey(key)) for key in range(STUDENTS_NUM)])
    # S, S_props = ([deepcopy(T0) , deepcopy(T0)]) , ([T0_props, T0_props])
    # S, S_props, student_hmm = add_ring(initial_probs, transition_matrix, emission_means, emissions_cov)


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

    
    'Generate datasets'
    gdata   = lambda params, key: generate_data_from_model(HMM, params, jr.PRNGKey(key), NUM_TRIALS, NUM_TIMESTEPS)
    _, T0_emissions_train           = gdata(T0, 1)
    T0_states, T0_emissions_test    = gdata(T0, 100)


    'Train'
    fit = lambda hmm_class, params, props, emissions : zip(*[hmm_class.fit_sgd(param, prop, emissions) for param, prop in zip(*[params, props])])
    fit = lambda hmm_class, params, props, emissions : hmm_class.fit_sgd(params, props, emissions)

    evaluate_func = lambda hmm_class : vmap(hmm_class.marginal_log_prob, [None, 0], 0) #evaluate
    ev = lambda hmm, features, test: (evaluate_func(hmm)(features, test)).mean() #eval_true


    like = lambda model : float((ev(student_hmm, model, T0_emissions_test)-base(T0_emissions_train, T0_emissions_test))/(ev(HMM, T0, T0_emissions_test)-base(T0_emissions_train, T0_emissions_test)))
    like = lambda model : float(ev(student_hmm, model, T0_emissions_test))
    liketrain = lambda model : float((ev(student_hmm, model, T0_emissions_train)-base(T0_emissions_train, T0_emissions_train))/(ev(HMM, T0, T0_emissions_train)-base(T0_emissions_train, T0_emissions_train)))
    liketrain = lambda model : float(ev(student_hmm, model, T0_emissions_train))
    

    # print(f'iteration test \n', like(S))
    # print(f'iteration train \n', liketrain(S))

    



    # S0, _   = fit(student_hmm, S, S_props, T0_emissions_train)
    # S00, _  = fit(student_hmm, S0, S_props, T0_emissions_train)



    # print(float((ev(student_hmm, S[0], T0_emissions_test)-base(T0_emissions_train, T0_emissions_test))/(ev(HMM, T0, T0_emissions_test)-base(T0_emissions_train, T0_emissions_test))))
    # print(float((ev(student_hmm, S0[0], T0_emissions_test)-base(T0_emissions_train, T0_emissions_test))/(ev(HMM, T0, T0_emissions_test)-base(T0_emissions_train, T0_emissions_test))))
    

    for i in range(1):
    #     maxtrain = -np.inf
    #     maxtest = -np.inf
    #     # for model in S:
    #     #     l = like(model)
    #     #     lt = liketrain(model)
    #     #     maxtest = l if l > maxtest else maxtest
    #     #     maxtrain = lt if lt > maxtrain else maxtrain
    #     print(f'iteration {i} test \n', maxtest)
    #     print(f'iteration {i} train \n', maxtrain)
    #     # print(f'iteration {i} test \n', like(S00))
    #     # print(f'iteration {i} train \n', liketrain(S00))
        print('train = ', ev(student_hmm, S, T0_emissions_train))
        print('test = ', ev(student_hmm, S, T0_emissions_test))
        S, _  = fit(student_hmm, S, S_props, T0_emissions_train)


    backup_hmm(S, S_props, filename="test.pkl")

    params, props = restore_hmm(filename="test.pkl")

    print(S)
    print(S_props)
    # print('train after backup = ', ev(student_hmm, params, T0_emissions_train))
    # print('test after backuo = ', ev(student_hmm, params, T0_emissions_test))
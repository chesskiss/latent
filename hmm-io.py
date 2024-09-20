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
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

from dynamax.hidden_markov_model import GaussianHMM
from dynamax.hidden_markov_model import DiagonalGaussianHMM
from dynamax.hidden_markov_model import SphericalGaussianHMM
from dynamax.hidden_markov_model import SharedCovarianceGaussianHMM
from dynamax.utils.plotting import CMAP, COLORS, white_to_color_cmap
from sklearn.preprocessing import StandardScaler


from sklearn.mixture import GaussianMixture 


#TODO normalize rows/columns only, depending on j/i
def normalize(A):
    A /= A.sum(axis=1, keepdims=True) #Normalize rows
    A = A.at[-1].set(1 - jnp.sum(A[:-1], axis=0)) #last row is defined
    A = A.at[:, -1].set(1 - jnp.sum(A[:, :-1], axis=1)) #last column is defined
    return A

def perturbation(perturbation_num, epsilon, initial_probs, transition_matrix, emissions_means, emissions_cov):
    key = jr.PRNGKey(perturbation_num)
    initial_probs += jr.uniform(key, shape=(true_num_states,))
    initial_probs = initial_probs / jnp.sum(initial_probs)
    
    rn  = lambda x : np.random.uniform(-x, x)
    p   = rn(epsilon) * jnp.eye(true_num_states) \
        + rn(epsilon) * jnp.roll(jnp.eye(true_num_states), 1, axis=1) \
        + rn(epsilon) / true_num_states
    transition_matrix   += p
    transition_matrix   = jnp.clip(transition_matrix, 0, None) # Ensures non-negativity
    transition_matrix   = normalize(transition_matrix)

    emissions_means += np.random.normal(-epsilon, epsilon, emission_means.shape) 

    emissions_cov   += np.random.normal(0, epsilon) * jnp.eye(EMISSION_DIM)[None, :, :]

    return initial_probs, transition_matrix, emission_means, emissions_cov

    

def initial(): 
    'Specify initial parameters of the HMM' # TODO change to if Model is not set. if set - transition matrix = model.transition, etc.
    key = jr.PRNGKey(0)  # Random seed
    initial_probs = jr.uniform(key, shape=(true_num_states,),  minval=0)
    initial_probs = initial_probs / jnp.sum(initial_probs)

    # initial_probs = jnp.ones(true_num_states) / true_num_states

    rn = lambda x : np.random.uniform(-x, x)
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



def plot_m_arr(hmms, EMISSION_DIM, states):
    fig, axs = plt.subplots(len(hmms), 3)

    # fig = plt.figure()
    # axs = fig.add_subplot(111, projection='3d')

    for i, hmm_matrix in enumerate(hmms):
        ######## Tranisition matrix Aij ########
        A = hmm_matrix.transitions.transition_matrix

        im = axs[i,0].imshow(A) #(x, y, z)
        fig.colorbar(im, ax= axs[i,0], label='Amplitude')
        axs[i,0].set_title('Tranisition matrix Aij')

        ######## Emmission matrix Bij ########
        B = hmm_matrix.emissions.means


        im = axs[i,1].imshow(B)
        fig.colorbar(im, ax= axs[i,1], label='Amplitude')
        axs[i,1].set_title('Emmission matrix Bij')


        ######## Initial states distribution ########
        initial_dist = hmm_matrix.initial.probs
        states = range(len(initial_dist))

        axs[i,2].bar(states, initial_dist)
        axs[i,2].set_title('Initial distributions')

        ########################################
    # Adjust layout to prevent overlap
    fig.tight_layout()
    fig.savefig("hmm-params.png")

# Helper functions for plotting
def plot_gaussian_hmm(hmm, params, emissions, states,  title="Emission Distributions", alpha=0.25):
    lim = 1.1 * abs(emissions).max()
    XX, YY = jnp.meshgrid(jnp.linspace(-lim, lim, 100), jnp.linspace(-lim, lim, 100))
    grid = jnp.column_stack((XX.ravel(), YY.ravel()))

    plt.figure()
    for k in range(hmm.num_states):
        lls = hmm.emission_distribution(params, k).log_prob(grid)
        plt.contour(XX, YY, jnp.exp(lls).reshape(XX.shape), cmap=white_to_color_cmap(COLORS[k]))
        plt.plot(emissions[states == k, 0], emissions[states == k, 1], "o", mfc=COLORS[k], mec="none", ms=3, alpha=alpha)

    plt.plot(emissions[:, 0], emissions[:, 1], "-k", lw=1, alpha=alpha)
    plt.xlabel("$y_1$")
    plt.ylabel("$y_2$")
    plt.title(title)
    plt.gca().set_aspect(1.0)
    plt.tight_layout()
    plt.savefig('Gaussian HMM emissions.png')
    plt.show()  # Ensure the plot is displayed


def plot_gaussian_hmm_data(hmm, params, emissions, states, xlim=None):
    NUM_TIMESTEPS = len(emissions)
    EMISSION_DIM = hmm.EMISSION_DIM
    means = params.emissions.means[states]
    lim = 1.05 * abs(emissions).max()

    # Plot the data superimposed on the generating state sequence
    fig, axs = plt.subplots(EMISSION_DIM, 1, sharex=True)
    
    for d in range(EMISSION_DIM):    
        axs[d].imshow(states[None, :], aspect="auto", interpolation="none", cmap=CMAP,
                      vmin=0, vmax=len(COLORS) - 1, extent=(0, NUM_TIMESTEPS, -lim, lim))
        axs[d].plot(emissions[:, d], "-k")
        axs[d].plot(means[:, d], ":k")
        axs[d].set_ylabel("$y_{{t,{} }}$".format(d+1))
        
    if xlim is None:
        plt.xlim(0, NUM_TIMESTEPS)
    else:
        plt.xlim(xlim)

    axs[-1].set_xlabel("time")
    axs[0].set_title("Simulated data from an HMM")
    plt.tight_layout()
    plt.show()  # Ensure the plot is displayed



NUM_TRAIN_BATCHS  = 3
NUM_TEST_BATCHS    = 1

NUM_EPOCHS          = 100000
NUM_TIMESTEPS       = 100
NUM_TRIALS          = 1000
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
    S0, S0_props    = hmm.initialize(jr.PRNGKey(0))

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

    S_l, S_l_props= hmm_n.initialize(jr.PRNGKey(0))


    'Generate datasets'
    gdata   = lambda params, key: generate_data_from_model(hmm, params, jr.PRNGKey(key), NUM_TRIALS, NUM_TIMESTEPS)
    _, T0_emissions_train           = gdata(T0, 1)
    T0_states, T0_emissions_test    = gdata(T0, 1000)
    _, T1_emissions_train           = gdata(T1, 1)
    T1_states, T1_emissions_test    = gdata(T1, 1000)
    teachers = [[T0, T0_emissions_train, T0_emissions_test], [T1, T1_emissions_train, T1_emissions_test], [T0, T0_emissions_train, T0_emissions_train]]


    # Plot emissions and true_states in the emissions plane
    # plot_gaussian_hmm(hmm, T0, T0_emissions_test[0], T0_states[0], title="True HMM emission distribution")

    # Plot emissions vs. time with background colored by true state
    # plot_gaussian_hmm_data(hmm, T0, T0_emissions_test[0], T0_states[0])

    # hmm.fit_em(S0, S0_props, T0_emissions_train)
    print('shape = checkpoint', T0_emissions_train.shape)

    #TODO - sum over all 100x100x3 ? over all 3 emissions dimensions?
    'Train'
    fit         = lambda hmm_class, params, props, emissions : hmm_class.fit_em(params, props, emissions) #TODO why it doesn't find it again?? Add to instructions to VIB! also add NUM_EPOCHS=NUM_EPOCHS
    S0, losses  = fit(hmm, S0, S0_props, T0_emissions_train)
    S1, _       = fit(hmm, S0, S0_props, T1_emissions_train)
    S01, _      = fit(hmm, S0, S0_props, T1_emissions_train)
    S01, _      = fit(hmm, S1, S0_props, T1_emissions_train)
    T01, _      = fit(hmm, T0, T0_props, T1_emissions_train)
    S_l1, _     = fit(hmm_n, S_l, S_l_props, T0_emissions_train)

    evaluate_func = lambda hmm_class : vmap(hmm_class.marginal_log_prob, [None, 0], 0) #evaluate
    ev = lambda hmm, features, test: (evaluate_func(hmm)(features, test)).mean() #eval_true

    params = [
        ["T0 (Ground truth)",   T0 , hmm],
        ["T1 (Perturbated)",    T1 , hmm],
        ["S0 (initial student)",S0 , hmm],
        ["S0",                  S0 , hmm],
        ["S1",                  S1 , hmm],
        ["S01",                 S01, hmm],
        ["S01",                 S01, hmm],
        ["True trained",        T01, hmm],
        ["S_l",                 S_l, hmm_n],
        ["S_l1",                S_l1, hmm_n]
    ] #TODO remove initial student if less than 0/baseline or certain value.
    results = {"Likelihood over" : ["T0", "T1", "over train T0"]}

    results_unormalized = {"Likelihood over" : ["T0", "T1", "over train T0"]}

    removed = []

    for T, train, test in teachers:
            print(f'base  = {base(train, test)}')
            # print(f' init = {ev(hmm,S0,test)}')
            # print(f' S0 = {ev(hmm, S0, test)}')
            # print(f' T = {ev(hmm, T, test)}')
            # print((ev_l(S_l1, test, hmm_n)-base(train, test))/(ev(T, test)-base(train, test)))
    # results['base'] = [base(train, test)  for T, train, test in teachers]

    print(f'value ' )

    for key, model, hmm_type in params:
        results[key] = [(ev(hmm_type, model, test)-base(train, test))/(ev(hmm, T, test)-base(train, test)) for T, train, test in teachers] #TODO change it back to minus
        results_unormalized[key] = [ev(hmm_type, model, test) for _, _, test in teachers] 
        if max(results[key])<0:
            del results[key]
            removed.append(key)

    df1 = pd.DataFrame(results)
    df2 = pd.DataFrame(results_unormalized)

    df = pd.concat([df1, df2])

    # (X - E(X))/σ(X)
    '''numeric_columns = df.select_dtypes(include=[np.number]).columns
    numeric_data = df[numeric_columns]
    print(numeric_columns)

    std_scale = StandardScaler()
    scaled_data = std_scale.fit_transform(numeric_data)

    df[numeric_columns] = scaled_data'''

    print(f'\nRemoved models with low performance: {removed}')
    # rownames(df) <- c("Likelihood") 
    print(df)
    df.to_csv('Params likelihood.csv', sep='\t')


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
1. Check dynamax and Claude for how to pertub teachers properly (so the likelihood won't surpass the teacher? Or perhaps it is ok)
1.1 Evaluate all on unseen teacher T3
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









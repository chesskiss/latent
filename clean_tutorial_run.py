import dynamax
from macros import *

from functools import partial

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax
from jax import vmap

from sklearn.mixture import GaussianMixture 
from dynamax.hidden_markov_model import CategoricalHMM


def likelihood(student_type, student_params, teacher, teacher_train_obs, test_obs):
    baseline_model = GaussianMixture(n_components=1)
    base = lambda train,test: baseline_model.fit(
            train.reshape(-1, EMISSION_DIM)
            ).score_samples(
                test.reshape(-1, EMISSION_DIM)
            ).reshape(NUM_TRIALS,NUM_TIMESTEPS).sum(axis=1).mean(axis=0)
    evaluate_func = lambda hmm_class : vmap(hmm_class.marginal_log_prob, [None, 0], 0) #evaluate
    ev = lambda hmm, features, test: (evaluate_func(hmm)(features, test)).mean() #eval_true
    print(f'Base liklihood: {base(teacher_train_obs, test_obs)}')
    print(f'Teacher liklihood: {ev(HMM, teacher, test_obs)}')
    print(f'Student liklihood: {ev(student_type, student_params, test_obs)}')
    return float((ev(student_type, student_params, test_obs)-base(teacher_train_obs, test_obs))/(ev(HMM, teacher, test_obs)-base(teacher_train_obs, test_obs)))




  # only one die is rolled at a time
num_classes = 6     # each die has six faces

initial_probs = jnp.array([0.5, 0.5])
transition_matrix = jnp.array([[0.95, 0.05], 
                               [0.10, 0.90]])
emission_probs = jnp.array([[1/6,  1/6,  1/6,  1/6,  1/6,  1/6],    # fair die
                            [1/10, 1/10, 1/10, 1/10, 1/10, 5/10]])  # loaded die


# Construct the HMM
hmm = CategoricalHMM(TRUE_NUM_STATES, EMISSION_DIM, num_classes)

# Initialize the parameters struct with known values
true_params, true_props = hmm.initialize(initial_probs=initial_probs,
                           transition_matrix=transition_matrix,
                           emission_probs=emission_probs.reshape(TRUE_NUM_STATES, EMISSION_DIM, num_classes))


num_batches = 5
num_timesteps = 5000
hmm = CategoricalHMM(TRUE_NUM_STATES, EMISSION_DIM, num_classes)

batch_states, batch_emissions = \
    vmap(partial(hmm.sample, true_params, num_timesteps=num_timesteps))(
        jr.split(jr.PRNGKey(42), num_batches))

print(f"batch_states.shape:    {batch_states.shape}") 
print(f"batch_emissions.shape: {batch_emissions.shape}") 



hmm = CategoricalHMM(TRUE_NUM_STATES, EMISSION_DIM, num_classes,
                     transition_matrix_stickiness=10.0)

key = jr.PRNGKey(0)
student_params, props = hmm.initialize(key)

fbgd_key = jr.PRNGKey(0)
fbgd_params, fbgd_losses = hmm.fit_sgd(true_params, 
                                       true_props, 
                                       batch_emissions, 
                                       optimizer=optax.sgd(learning_rate=0.025, momentum=0.95),
                                       batch_size=num_batches, 
                                       num_epochs=500, 
                                       key=fbgd_key)


sgd_key = jr.PRNGKey(0)


sgd_params, sgd_losses = hmm.fit_sgd(true_params, 
                                    true_props, 
                                    batch_emissions, 
                                    optimizer=optax.sgd(learning_rate=0.025, momentum=0.95),
                                    #  batch_size=2, 
                                    num_epochs=500, 
                                    key=sgd_key)


l_before = likelihood(hmm, student_params, true_params, batch_emissions, batch_emissions)
l_after = likelihood(hmm, sgd_params, true_params, batch_emissions, batch_emissions)    


print(f'before : {l_before} \n\n after: {l_after}')


true_loss = vmap(partial(hmm.marginal_log_prob, true_params))(batch_emissions).sum()
true_loss += hmm.log_prior(true_params)
true_loss = -true_loss / batch_emissions.size


plt.plot(fbgd_losses, label="full batch GD")
plt.plot(sgd_losses, label="SGD (m.b. size=2)")
plt.axhline(true_loss, color='k', linestyle=':', label="True Params")
plt.legend()
plt.xlim(-10, 500)
plt.xlabel("epoch")
plt.ylabel("loss")
_ = plt.title("Learning Curve Comparison")
plt.show()

from functools import partial
from jax import vmap
import jax.numpy as jnp
import jax.random as jr
from hmmST import likelihood
from macros import *
import optax

data_sample = lambda model, params, key:  model.sample(params, jr.PRNGKey(key), NUM_TIMESTEPS)

fit = lambda hmm_class, params, props, emissions : hmm_class.fit_sgd(params, props, emissions, optimizer=optax.sgd(learning_rate=1e-3))

def evaluate_model(hmm, model, test_data):
    true_loss = vmap(partial(hmm.marginal_log_prob, model))(test_data).sum()
    true_loss += hmm.log_prior(model)
    true_loss = -true_loss / test_data.size
    return true_loss

def norm_loss(hmm, model, test_data, true_hmm, true_model):
    return (evaluate_model(hmm, model, test_data)-2000) / (evaluate_model(true_hmm, true_model, test_data)-2000)


evaluate_func = lambda hmm_class : vmap(hmm_class.marginal_log_prob, [None, 0], 0) #evaluate
ev = lambda hmm, features, test: (evaluate_func(hmm)(features, test)).mean() #eval_true


hmm = GaussianHMM(TRUE_NUM_STATES, EMISSION_DIM)
T, _ = hmm.initialize(jr.PRNGKey(10))
S, S_props  = hmm.initialize(jr.PRNGKey(0))

_, train_data  = data_sample(hmm, T, 0)
_, test_data    = data_sample(hmm, T, 1)


# NUM_TRAIN_BATCHS = 5
# NUM_TIMESTEPS = 10

# states_num, train_data = vmap(partial(hmm.sample, T, num_timesteps=NUM_TIMESTEPS))(
#         jr.split(jr.PRNGKey(42), NUM_TRAIN_BATCHS))

# _, test_data = \
#     vmap(partial(hmm.sample, T, num_timesteps=NUM_TIMESTEPS))(
#         jr.split(jr.PRNGKey(99), 1))




opt = None
for i in range(10):
    # print(f'iteration {i}, train: ', ev(hmm, S, train_data))
    # print(f'iteration {i}, test: ', ev(hmm, S, test_data))
    print(f'iteration {i}, likelihood: ', likelihood(hmm, S, T, train_data, train_data))

    # print(f'iteration {i}, train: ', evaluate_model(hmm, T, test_data))
    # print(f'iteration {i}, test: ', loss())
    # S, _ = fit(hmm, S, S_props, train_data)
    sgd_key = jr.PRNGKey(0)
    S, opt, sgd_losses = hmm.fit_sgd(S, 
                                        S_props, 
                                        train_data, 
                                        # optimizer=optax.adam(learning_rate=0.025, momentum=0.95),
                                        batch_size=2, 
                                        num_epochs=500,
                                        init_opt_state = opt,
                                        key=sgd_key)
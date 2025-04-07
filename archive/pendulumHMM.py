# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.integrate import solve_ivp

# # import jax.numpy as jnp
# # import matplotlib.pyplot as plt
# # from jax import vmap 
# # import jax.random as jr
# # from functools import partial

# # from dynamax.hidden_markov_model import GaussianHMM

# from sktime.annotation.hmm_learn import GaussianHMM 
# from sktime.annotation.datagen import piecewise_normal 


####
import subprocess
import sys

try:
    import dynamax
except ModuleNotFoundError:
    print('installing dynamax')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'dynamax[notebooks]'])
    import dynamax

from functools import partial

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from jax import vmap

from sktime.annotation.hmm_learn import GaussianHMM 
from dynamax.hidden_markov_model import DiagonalGaussianHMM
from dynamax.hidden_markov_model import SphericalGaussianHMM
from dynamax.hidden_markov_model import SharedCovarianceGaussianHMM
from dynamax.utils.plotting import CMAP, COLORS, white_to_color_cmap



def plot_movement(time, theta, omega):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(time, theta)
    plt.title('Pendulum Angle (Theta)')
    plt.xlabel('Time [s]')
    plt.ylabel('Angle [rad]')

    plt.subplot(1, 2, 2)
    plt.plot(time, omega)
    plt.title('Pendulum Angular Velocity (Omega)')
    plt.xlabel('Time [s]')
    plt.ylabel('Angular Velocity [rad/s]')

    plt.tight_layout()
    plt.show()



# Define pendulum dynamics
def pendulum_dynamics(theta, omega):
    x = [theta, omega]
    # Time span for the simulation
    t_span = (0, 10)  # 10 seconds
    t_eval = np.linspace(t_span[0], t_span[1], 500)  # evaluation points

    g = 9.81  # Gravity acceleration (m/s^2)
    L = 1.0   # Length of the pendulum (m)
    A = np.array([[0, 1], [-g/L, 0]])  # State-space representation
    solution = solve_ivp(lambda t, u: A @ u, t_span, x, t_eval=t_eval, vectorized=True)  # Solves equation
    theta_out = solution.y[0]
    omega_out = solution.y[1]
    time = solution.t
    return theta_out, omega_out, time



# Generate pendulum data for each set of initial conditions
def generate_pendulum_data(initial_conditions):
    data = []
    for theta_x, omega_x in initial_conditions:
        theta_y, omega_y, time_y = pendulum_dynamics(theta_x, omega_x)
        data.append((theta_y, omega_y, time_y))
    return np.array(data)



# # Generate vectors of initial conditions
# theta_x_values = np.linspace(0, np.pi / 2, 5)  # 5 values from 0 to 90 degrees
# omega_x_values = np.linspace(-2, 2, 5)        # 5 values from -2 to 2 rad/s

# # Create a grid of initial conditions
# initial_conditions = [(theta, omega) for theta in theta_x_values for omega in omega_x_values]

# # Generate train and test data using the grid of initial conditions
# train_emissions = generate_pendulum_data(initial_conditions) # train_data
# test_emissions  = generate_pendulum_data(initial_conditions) # test_data

# # Ensure correct shape for HMM
# num_train_batches = len(train_emissions)
# num_test_batches = len(test_emissions)
# train_emissions = train_emissions.reshape(num_train_batches, -1, 3)
# test_emissions = test_emissions.reshape(num_test_batches, -1, 3)



# emission_dim = 3 # output dim (theta, omega, t)


# def cross_validate_model(model, key, num_iters=100):
#     # Initialize the parameters using K-Means on the full training set
#     params, props = model.initialize(key=key, method="kmeans", emissions=train_emissions)
    
#     # Split the training data into folds.
#     # Note: this is memory inefficient but it highlights the use of vmap.
#     folds = jnp.stack([
#         jnp.concatenate([train_emissions[:i], train_emissions[i+1:]])
#         for i in range(num_train_batches)
#     ])
    
#     def _fit_fold(y_train, y_val):
#         fit_params, train_lps = model.fit_em(params, props, y_train, 
#                                              num_iters=num_iters, verbose=False)
#         return model.marginal_log_prob(fit_params, y_val)

#     val_lls = vmap(_fit_fold)(folds, train_emissions)
#     return val_lls.mean(), val_lls

# # Run cross validation fucntion
# # Make a range of Gaussian HMMs
# all_num_states = list(range(2, 10))
# test_hmms = [GaussianHMM(num_states, emission_dim, transition_matrix_stickiness=10.) 
#           for num_states in all_num_states]
# results = []
# for test_hmm in test_hmms:
#     print(f"fitting model with {test_hmm.num_states} states")
#     results.append(cross_validate_model(test_hmm, jr.PRNGKey(0)))
    
# avg_val_lls, all_val_lls = tuple(zip(*results))




num_train_batches = 3
num_test_batches = 1
num_timesteps = 100

# Make an HMM and sample data and true underlying states
true_num_states = 3
emission_dim = 2
hmm = GaussianHMM(true_num_states, emission_dim)

# Specify parameters of the HMM
initial_probs = jnp.ones(true_num_states) / true_num_states
transition_matrix = 0.80 * jnp.eye(true_num_states) \
    + 0.15 * jnp.roll(jnp.eye(true_num_states), 1, axis=1) \
    + 0.05 / true_num_states
emission_means = jnp.column_stack([
    0.1*jnp.cos(jnp.linspace(0, 2 * jnp.pi, true_num_states + 1))[:-1],
    0.*jnp.sin(jnp.linspace(0, 2 * jnp.pi, true_num_states + 1))[:-1],
    jnp.zeros((true_num_states, emission_dim - 2)),
    ])
emission_covs = jnp.tile(0.1**2 * jnp.eye(emission_dim), (true_num_states, 1, 1))
print(emission_means)
        
true_params, _ = hmm.initialize(initial_probs=initial_probs,
                                transition_matrix=transition_matrix,
                                emission_means=emission_means,
                                emission_covariances=emission_covs)




# #testing input
# theta = np.pi / 4
# omega = 1.0

# import pandas as pd

# # Create a DataFrame with 'theta' and 'omega' columns

# data = piecewise_normal( 
#    means=[2, 4, 1], lengths=[10, 35, 40], random_state=7
#    ).reshape((-1, 1))
# hmm = GaussianHMM() 


# true_params, _ = hmm.initialize()

# data = hmm.sample(true_params, 10) #10=time stamos 

# # model = model.fit(data) 

# # data = pd.DataFrame({'theta': [np.pi / 4], 'omega': [1.0]})

# labeled_data = hmm.smoother(true_params, data)

# print(labeled_data)

# # plot_movement(time, theta, omega)

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from dynamax.utils.plotting import CMAP, COLORS, white_to_color_cmap

from hmmST import *



def visualize(df):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.cm.rainbow(np.linspace(0, 1, len(df.columns)))

    for col, color in zip(df.columns, colors):
        x = df.loc['T0', col]
        y = df.loc['T1', col]
        z = df.loc['T2', col]
        
        # Check if we have a list of points
        if isinstance(x, list):
            for i in range(len(x)):
                ax.scatter(x[i], y[i], z[i], c=[color], marker='o', s=50, label=col if i == 0 else "")
        else:
            ax.scatter(x, y, z, c=[color], marker='o', s=50, label=col)

    ax.set_xlabel('Performance on T0')
    ax.set_ylabel('Performance on T1')
    ax.set_zlabel('Performance on T2')
    ax.set_title(f'Likelihood Plot\n Students num: {STUDENTS_NUM} Emissions Dim: {EMISSION_DIM} True num states: {TRUE_NUM_STATES}')

    # Put legend outside the plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig('Likelihoods graph.png', bbox_inches='tight')
    plt.show()




# def visualize(df):
#     # df.set_index('Likelihood over', inplace=True) # Set 'Likelihood over' as the index

#     # fig = plt.figure()
#     # ax = fig.add_subplot(111, projection='3d')

#     # x = df.loc['T0']
#     # y = df.loc['T1']
#     # z = df.loc['T2']

#     # ax.scatter(x, y, z)

#     # ax.set_xlabel('T0')
#     # ax.set_ylabel('T1')
#     # ax.set_zlabel('T2')
#     # ax.set_title('3D Likelihood Plot')

#     # plt.savefig('Likelihoods graph.png')
#     # plt.show()

#     df.set_index('Likelihood over', inplace=True)  # Set 'Likelihood over' as the index
    
#     fig = plt.figure(figsize=(12, 8))
#     ax = fig.add_subplot(111, projection='3d')
    
#     # Separate columns that begin with 'T' and those that don't
#     t_cols = [col for col in df.columns if col.startswith('T')]
#     s_cols = [col for col in df.columns if not col.startswith('T')]
    
#     # Plot students
#     x = df.loc['T0', s_cols]
#     y = df.loc['T1', s_cols]
#     z = df.loc['T2', s_cols]
#     ax.scatter(x, y, z, c='b', marker='o', s=50, label='S group')
    
#     # Plot teachers
#     s = df.loc['T0', t_cols]
#     u = df.loc['T1', t_cols]
#     v = df.loc['T2', t_cols]
#     ax.scatter(s, u, v, c='r', marker='x', s=100, label='T group')
    
#     ax.set_xlabel('Performance on T0')
#     ax.set_ylabel('Performance on T1')
#     ax.set_zlabel('Performance on T2')
#     ax.set_title('3D Likelihood Plot')
    
#     ax.legend()
    
#     plt.savefig('Likelihoods graph.png')
#     plt.show()




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
    plt.show()  



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
    plt.show()
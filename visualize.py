import numpy as np
import jax.numpy as jnp
import networkx as nx
import matplotlib.pyplot as plt
from dynamax.utils.plotting import CMAP, COLORS, white_to_color_cmap
import pandas as pd
import os
import json
from hmmST import *
from macros import *


'Plot with X = epochs, Y= decoding per teacher and likelihood'
def plot_decodingEpochs(train_likelihoods, test_likelihoods, decodingST, decodingTS, csv_file_name='decoding_data'):
    csv_path = f'{csv_file_name}.csv'
    
    if train_likelihoods is None:
        df = pd.read_csv(csv_path)
        
        # Ensure required columns exist
        required_columns = {'epoch', 'train_likelihoods', 'test_likelihoods', 'decodingST', 'decodingTS'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"CSV file missing required columns: {required_columns - set(df.columns)}")

        # Determine number of epochs and students dynamically
        num_epochs = df['epoch'].nunique()
        num_students = len(json.loads(df['train_likelihoods'].iloc[0]))  # Extract from first row
        num_teachers = len(json.loads(df['train_likelihoods'].iloc[0])[0])  # Extract from first student's data

        data = {'epoch': [], 'train_likelihoods': [], 'test_likelihoods': [], 'decodingST': [], 'decodingTS': []}

        # Update loading logic
        decodingST = np.zeros((num_epochs, num_students, num_teachers))
        decodingTS = np.zeros((num_epochs, num_students, num_teachers))
        train_likelihoods = np.zeros((num_epochs, num_students, num_teachers))
        test_likelihoods = np.zeros((num_epochs, num_students, num_teachers))

        # Populate arrays
        for _, row in df.iterrows():
            e = int(row['epoch'])
            train_likelihood_matrix = np.array(json.loads(row['train_likelihoods']))
            test_likelihood_matrix = np.array(json.loads(row['test_likelihoods']))
            decodingST_matrix = np.array(json.loads(row['decodingST']))
            decodingTS_matrix = np.array(json.loads(row['decodingTS']))

            train_likelihoods[e] = train_likelihood_matrix
            test_likelihoods[e] = test_likelihood_matrix
            decodingST[e] = decodingST_matrix
            decodingTS[e] = decodingTS_matrix

    elif os.path.exists(csv_path):
        existing_files = [f for f in os.listdir('.') if f.startswith(f'{csv_file_name}_') and f.endswith('.csv')]
        next_index = len(existing_files) + 1
        csv_path = f'{csv_file_name}_{next_index}_{NUM_EPOCHS}Epochs.csv'

        # Compute values dynamically and save them
        data = {'epoch': [], 'train_likelihoods': [], 'test_likelihoods': [], 'decodingST': [], 'decodingTS': []}

        for e in range(NUM_EPOCHS):
            data['epoch'].append(e)
            data['train_likelihoods'].append(json.dumps(train_likelihoods[e]))
            data['test_likelihoods'].append(json.dumps(test_likelihoods[e]))
            data['decodingST'].append(json.dumps(decodingST[e]))
            data['decodingTS'].append(json.dumps(decodingTS[e]))

        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)

    else:
        # Compute values dynamically and save them

        data = {'epoch': [], 'train_likelihoods': [], 'test_likelihoods': [], 'decodingST': [], 'decodingTS': []}

        for e in range(NUM_EPOCHS):
            data['epoch'].append(e)
            data['train_likelihoods'].append(json.dumps(train_likelihoods[e]))
            data['test_likelihoods'].append(json.dumps(test_likelihoods[e]))
            data['decodingST'].append(json.dumps(decodingST[e]))
            data['decodingTS'].append(json.dumps(decodingTS[e]))

        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)


    epochs = range(len(train_likelihoods))
    num_students = len(train_likelihoods[0])
    num_teachers = train_likelihoods.shape[2] if isinstance(train_likelihoods, np.ndarray) else len(train_likelihoods[0][0])

    # Plotting
    fig, axes = plt.subplots(3, 3, figsize=(12, 12), sharey='row')
    titles = ['T0', 'T1', 'T2']
    colors = plt.cm.viridis(np.linspace(0, 1, num_students))  # Use color map for students

    # Plot likelihoods (row 1)
    for t in range(num_teachers):
        for s in range(num_students):
            axes[0, t].plot(epochs, [l[s][t] for l in train_likelihoods], label=f'Train Student {s}', color=colors[s], linestyle='--')
            axes[0, t].plot(epochs, [l[s][t] for l in test_likelihoods], label=f'Test Student {s}', color=colors[s], linestyle='-')
        axes[0, t].set_title(f'Likelihood - {titles[t]}')
        axes[0, t].set_xlabel('Epochs')
        axes[0, t].set_ylabel('Likelihood')
        axes[0, t].legend()

    # Plot decoding T -> S (row 2)
    for t in range(num_teachers):
        for s in range(num_students):
            axes[1, t].plot(epochs, [d[s][t] for d in decodingST], label=f'Student {s}', color=colors[s])
        axes[1, t].set_title(f'Decoding T{t} → S')
        axes[1, t].set_xlabel('Epochs')
        axes[1, t].set_ylabel('Error')
        axes[1, t].legend()

    # Plot decoding S -> T (row 3)
    for t in range(num_teachers):
        for s in range(num_students):
            axes[2, t].plot(epochs, [d[s][t] for d in decodingTS], label=f'Student {s}', color=colors[s])
        axes[2, t].set_title(f'Decoding S → T{t}')
        axes[2, t].set_xlabel('Epochs')
        axes[2, t].set_ylabel('Error')
        axes[2, t].legend()

    plt.tight_layout()
    plt.savefig('./DecodingLikelihoodEpochs.png', dpi=300, bbox_inches='tight')
    plt.show()




'Visualize performances'
def performance_plot(df):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(df.columns)))
    
    pairs = [('T0', 'T1'), ('T0', 'T2'), ('T1', 'T2')]
    
    for idx, (x_label, y_label) in enumerate(pairs):
        ax = axs[idx]
        
        all_x = []
        all_y = []
        
        for col, color in zip(df.columns, colors):
            x_values = df.loc[x_label, col]
            y_values = df.loc[y_label, col]
            
            
            # Plot each pair of points
            ax.scatter(x_values, y_values, c=[color], marker='o', s=50, label=col)
            
            all_x.extend(x_values)
            all_y.extend(y_values)
        
        ax.set_xlabel(f'Performance on {x_label}')
        ax.set_ylabel(f'Performance on {y_label}')
        ax.set_title(f'{x_label} vs {y_label}')
        
        # Set axis limits based on data range
        # ax.set_xlim(0, 1)
        # ax.set_ylim(0, 1)
        
        # Use scientific notation for axis labels if the range is large
        ax.ticklabel_format(style='sci', scilimits=(-2,2), axis='both')
    
    # Set a common title for all subplots
    fig.suptitle('Likelihood Plots')
    
    # Put legend outside the rightmost plot
    axs[-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('Likelihoods graphs.png', bbox_inches='tight')
    plt.show()



def performance_plot_3D(df):
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
    ax.set_title(f'Likelihood Plot\n Students num: {STUDENTS_NUM} Emissions Dim: {EMISSION_DIM} \
                 True num states: {TRUE_NUM_STATES} Range of S states: [{TRUE_NUM_STATES}, {MAX_S_STATE-1}]')

    # Put legend outside the plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig('Likelihoods graph.png', bbox_inches='tight')
    plt.show()


def transitions_plot(student, states):
    transition_matrix, state_visit_count = state_transition_counter(np.array(states),student.n_components)

    total_timesteps = states.shape[0] * states.shape[1]
    total_transitions = states.shape[0] * (states.shape[1]-1)
    plot_state_transition_graph(
        transition_matrix/total_transitions, 
        state_visit_count/total_timesteps, 
        edge_threshold=1e-2,
        spring_kwargs = cfg.spring_kwargs if hasattr(cfg,'spring_kwargs') else {},
        savepath = results_path+f'_histogram_graph.pdf'
)

'Visualize the datasets & emissions'
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



def state_transition_counter(state_sequence_array, M):
    # Initialize transition matrix and state visit count vector
    transition_matrix = np.zeros((M, M), dtype=int)
    state_visit_count = np.zeros(M, dtype=int)

    # Loop through each trial sequence
    for trial in state_sequence_array:
        # Flatten the array to ensure it is a 1D sequence of states
        states = trial.flatten()

        # Update the state visit counts
        for state in states:
            state_visit_count[state] += 1

        # Update the transition matrix for each consecutive state pair
        for i in range(len(states) - 1):
            current_state = states[i]
            next_state = states[i + 1]
            transition_matrix[current_state, next_state] += 1

    return transition_matrix, state_visit_count


def plot_state_transition_graph_old(transition_matrix, state_visit_count, savepath='state_transition_graph'):
    # Create a directed graph
    G = nx.DiGraph()

    # Number of states
    M = len(state_visit_count)

    # Add nodes with sizes proportional to state visit count
    for state in range(M):
        G.add_node(state, size=state_visit_count[state])

    # Add edges with weight proportional to transition counts
    max_transition = np.max(transition_matrix) if np.max(transition_matrix) > 0 else 1
    for from_state in range(M):
        for to_state in range(M):
            if transition_matrix[from_state, to_state] > 0:
                # Normalize width
                width = 5 * (transition_matrix[from_state, to_state] / max_transition)
                G.add_edge(from_state, to_state, weight=width)

    # Define the node sizes and edge widths
    node_sizes = [state_visit_count[state] * 100 for state in range(M)]
    edge_widths = [G[u][v]['weight'] for u, v in G.edges()]

    # Plot the graph using a circular layout
    pos = nx.circular_layout(G)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.8)

    # Draw edges with explicit color and width to avoid artifacts
    nx.draw_networkx_edges(
        G, pos,
        arrowstyle='->',
        arrowsize=10,
        width=edge_widths,
        edge_color='gray',  # Set a fixed color to avoid unwanted fills
        connectionstyle='arc3,rad=0.2',
    )

    # Remove axis and margins
    plt.gca().set_axis_off()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save the figure as PNG and PDF
    plt.savefig(f"{savepath}.png", format='png')
    plt.savefig(f"{savepath}.pdf", format='pdf')

    # Close the plot to free up resources
    plt.close()


# Example usage:

def plot_state_transition_graph(transition_matrix, initial_distribution, edge_threshold=0.05, savepath=None, seed=0):
    print('trans mat:', transition_matrix.shape, 'init dist:', initial_distribution.shape)
    # Create a directed graph
    plt.figure(figsize=(7, 6))
    G = nx.DiGraph()

    # Add nodes with color attribute
    for i in range(len(initial_distribution)):
        G.add_node(i, color=initial_distribution[i])

    # Add edges filtering by edge_threshold
    for i in range(transition_matrix.shape[0]):
        for j in range(transition_matrix.shape[1]):
            if transition_matrix[i, j] > edge_threshold:
                if G.has_edge(j, i):
                    # Add curved edges if bidirectional
                    G.add_edge(i, j, weight=transition_matrix[i, j], connectionstyle='arc3,rad=0.2')
                else:
                    # Straight edge if unidirectional
                    G.add_edge(i, j, weight=transition_matrix[i, j], connectionstyle='arc3,rad=0')

    # Define color map based on initial distribution
    color_map = [plt.cm.hot(G.nodes[i]['color']) for i in G.nodes]

    # Draw the graph layout
    pos = nx.spring_layout(G, seed=seed)
    edges = G.edges(data=True)

    # Manually set zorder using matplotlib's scatter for nodes
    ax = plt.gca()  # Get current axis

    node_sizes = 1e6 * initial_distribution
    node_collection = nx.draw_networkx_nodes(G, pos,
                                             node_size=np.sqrt(node_sizes))  # node_size=1400 #node_color=color_map,
    # node_collection.set_zorder(2)  # Set zorder manually for nodes

    # Draw labels and set their zorder manually
    # label_dict = nx.draw_networkx_labels(G, pos, font_color='white', font_size=30)
    # for label in label_dict.values():
    #     label.set_zorder(3)  # Manually set the zorder for each label

    # Edge width scaled by weight
    edge_widths = [d['weight'] * 100 for (_, _, d) in edges]

    # Draw edges manually using LineCollection from matplotlib for z-order control
    for (u, v, d) in edges:
        alpha_value = min(1.0, d['weight'] * 10)  # Scale alpha based on weight

        connectionstyle = d.get('connectionstyle', 'arc3,rad=0')
        # Use matplotlib to manually plot the edges
        line = nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], width=3,  # width=edge_widths.pop(0),
            connectionstyle=connectionstyle, arrowstyle='-|>', arrowsize=20,
            edge_color='gray', alpha=alpha_value
        )
        # You can access the created collection and adjust zorder if needed
        # if isinstance(line, list):
        #     for ln in line:
        #         ln.set_zorder(1)  # Set zorder for edges

    # Add edge labels (optional)
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}

    plt.axis('off')

    # plt.tight_layout()

    # Save the graph if savepath is provided
    if savepath:
        plt.savefig(savepath, transparent=True, dpi=300)

    plt.close()


def plot_state_transition_graph(transition_matrix, initial_distribution, edge_threshold=0.05, savepath=None,
                                spring_kwargs={}):
    print('trans mat:', transition_matrix.shape, 'init dist:', initial_distribution.shape)

    # Create a directed graph
    plt.figure(figsize=(7, 6))
    G = nx.DiGraph()

    # Add nodes with color attribute
    for i in range(len(initial_distribution)):
        G.add_node(i, color=initial_distribution[i])

    # Add edges filtering by edge_threshold
    for i in range(transition_matrix.shape[0]):
        for j in range(transition_matrix.shape[1]):
            if transition_matrix[i, j] > edge_threshold:
                if G.has_edge(j, i):
                    # Add curved edges if bidirectional
                    G.add_edge(i, j, weight=transition_matrix[i, j], connectionstyle='arc3,rad=0.2')
                else:
                    # Straight edge if unidirectional
                    G.add_edge(i, j, weight=transition_matrix[i, j], connectionstyle='arc3,rad=0')

    # Define color map based on initial distribution
    color_map = [plt.cm.hot(G.nodes[i]['color']) for i in G.nodes]

    # Draw the graph layout
    pos = nx.spring_layout(G, **spring_kwargs)
    edges = G.edges(data=True)

    # Manually set zorder using matplotlib's scatter for nodes
    ax = plt.gca()  # Get current axis

    node_sizes = 1e6 * initial_distribution

    # Edge width scaled by weight
    edge_weights = [d['weight'] for (_, _, d) in edges]

    # Normalize edge weights for color mapping (from 0 to 1)
    # max_weight = max(edge_weights)
    max_weight = 0.25
    # min_weight = min(edge_weights)
    min_weight = 0  # min(edge_weights)
    norm_edge_weights = [(w - min_weight) / (max_weight - min_weight) for w in edge_weights]
    norm_node_values = [(w - min_weight) / (max_weight - min_weight) for w in initial_distribution]

    # Create a colormap from white (low values) to black (high values)
    # edge_colors = [plt.cm.gray(1 - norm_w) for norm_w in norm_edge_weights]
    # edge_colors = [plt.cm.gray(0.2 + 0.8 * (1 - norm_w)) for norm_w in norm_edge_weights]
    edge_colors = [plt.cm.inferno_r(0.2 + 0.8 * (1 - norm_w)) for norm_w in norm_edge_weights]
    node_colors = [plt.cm.inferno_r(0.2 + 0.8 * (1 - norm_w)) for norm_w in norm_node_values]
    # edge_colors = [plt.cm.inferno(norm_w) for norm_w in norm_edge_weights]

    # Draw nodes with specified colours and sizes
    node_collection = nx.draw_networkx_nodes(G, pos, node_size=np.sqrt(node_sizes), node_color=node_colors)

    # Draw edges with the color based on normalized weight
    for (u, v, d), color in zip(edges, edge_colors):
        alpha_value = min(1.0, d['weight'] * 10)
        connectionstyle = d.get('connectionstyle', 'arc3,rad=0')
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)],
            # width=3,
            width=d['weight'] * 20,
            connectionstyle=connectionstyle,
            arrowstyle='-|>', arrowsize=50,  # d['weight']*200,
            edge_color=[color],
            # edge_color='grey',
            alpha=1  # alpha_value  # Use color from the grayscale colormap
        )

    plt.axis('off')

    # Save the graph if savepath is provided
    if savepath:
        plt.savefig(savepath, transparent=True, dpi=300)

    plt.show()
    plt.close()


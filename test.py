# import jax.numpy as jnp
# from jax import vmap

# vv = lambda x, y: jnp.vdot(x, y)  #  ([a], [a]) -> []
# mv = vmap(vv, (0, None), 0)      #  ([b,a], [a]) -> [b]      (b is the mapped axis)
# mm = vmap(mv, (None, 1), 1)      #  ([b,a], [a,c]) -> [b,c]  (c is the mapped axis)

# a = jnp.array([1,1,1])
# b = jnp.array([2,1,0])
# A = a[None]
# print(A)
# print(mv(A,b))


import pandas as pd
import numpy as np

def dframe(data, index=['T0', 'T1', 'T2']):
    result = {}
    for key, value in data.items():
        arr = np.array(value)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        result[key] = arr.T.tolist()
    
    df = pd.DataFrame(result, index=index)
    return df

# Test data
data = {
    'S0': [[0.9310199618339539, -487.2692565917969, -1222.5260009765625], 
           [0.9301355481147766, -470.9032897949219, -1179.9246826171875]],
    'S1': [[-10.34585189819336, 0.03304697945713997, -8.23599910736084], 
           [-10.432733535766602, 0.14426057040691376, -8.45174789428711]],
    'S00': [[0.9338423013687134, -484.57708740234375, -1217.4638671875], 
            [0.9307991862297058, -467.2206115722656, -1171.578857421875]],
    'S01': [[-10.307413101196289, 0.7140099406242371, -7.795291900634766], 
            [-10.386242866516113, 0.706073522567749, -7.8934407234191895]],
    'S11': [[-10.335633277893066, 0.19122131168842316, -8.180229187011719], 
            [-10.410856246948242, 0.1675550490617752, -8.368451118469238]],
    'S_l0': [[0.645107626914978, -462.34185791015625, -1166.82470703125]]
}

df = dframe(data)
# print(df)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def visualize(df):
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
            
            # Ensure x_values and y_values are lists
            if not isinstance(x_values, list):
                x_values = [x_values]
            if not isinstance(y_values, list):
                y_values = [y_values]
            
            # Plot each pair of points
            ax.scatter(x_values, y_values, c=[color], marker='o', s=50, label=col)
            
            all_x.extend(x_values)
            all_y.extend(y_values)
        
        ax.set_xlabel(f'Performance on {x_label}')
        ax.set_ylabel(f'Performance on {y_label}')
        ax.set_title(f'{x_label} vs {y_label}')
        
        # Set axis limits based on data range
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Use scientific notation for axis labels if the range is large
        ax.ticklabel_format(style='sci', scilimits=(-2,2), axis='both')
    
    # Set a common title for all subplots
    fig.suptitle('Likelihood Plots')
    
    # Put legend outside the rightmost plot
    axs[-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('Likelihoods graphs.png', bbox_inches='tight')
    plt.show()

# Visualize the data
df = pd.read_csv('Params likelihood.csv', index_col=0)

visualize(df)
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import ops
from scipy.integrate import solve_ivp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from sktime.annotation.hmm_learn import GaussianHMM 
from sktime.annotation.datagen import piecewise_normal 

# Define plot_movement function
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



# Define pendulum dynamics function
def pendulum_dynamics(theta, omega):
    g = 9.81  # Gravity acceleration (m/s^2)
    L = 1.0   # Length of the pendulum (m)
    
    # State-space representation
    A = np.array([[0, 1],
                  [-g/L, 0]])
    
    # Time span for the simulation
    t_span = (0, 10)  # 10 seconds
    t_eval = np.linspace(t_span[0], t_span[1], 500)  # evaluation points
    
    # Solve the differential equation
    solution = solve_ivp(lambda t, y: A @ y, t_span, [theta, omega], t_eval=t_eval)
    
    # Extract theta, omega, and time
    theta_out = solution.y[0]
    omega_out = solution.y[1]
    time = solution.t
    
    return theta_out, omega_out, time



def keras_models_build(theta_x, omega_x, theta_y, omega_y, time_y):
    # Reshape the inputs to match Keras requirements
    x = np.array([[theta_x, omega_x]])  # Shape: (1, 2)
    y = np.array([[theta_y, omega_y, time_y]])  # Shape: (1, 3)
    # Build the Keras model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(2,)),
        Dense(1500, activation='linear'),  # Output layer with 3 units (theta_pred, omega_pred, time_pred)x500 time steps
        Reshape((3, 500))  # Reshape the output to (1, 3, 500)
    ])
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    #Train the model
    history = model.fit(x, y, epochs=1000, verbose=1)
    model.save("model_checkpoint.keras")

def keras_model_predict(theta_test, omega_test):
    x_test = np.array([[theta_test, omega_test]]) 
    predicted_outputs = model.predict(x_test)

    time_out    = predicted_outputs[0][2]
    theta_out   = predicted_outputs[0][1]
    omega_out   = predicted_outputs[0][0]

    return time_out, theta_out, omega_out



# Input/features (Initial conditions)
theta_x = np.pi / 4  # Initial angle (rad)
omega_x = 1.0        # Initial angular velocity (rad/s)
theta_test, omega_test = np.pi/10, 0.1

# Ground truth
theta_true, omega_true, time_true = pendulum_dynamics(theta_x, omega_x)
plot_movement(time_true, theta_true, omega_true)


#keras_models_build(theta_x, omega_x, theta_y, omega_y, time_y) # Train only once
model = keras.models.load_model("model_checkpoint.keras")
# Evaluate the model on the initial conditions
time_out, theta_out, omega_out = keras_model_predict(theta_test, omega_test)
plot_movement(time_out, theta_out, omega_out)


#HMM
hmm = GaussianHMM(true_num_states, emission_dim, transition_matrix_stickiness=10.),


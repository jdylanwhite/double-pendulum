"""Simulate a double pendulum."""

# Import modules
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define the gravitational constant
g = 9.81

# Define the initial conditions
# theta1 and theta2 are the initial angles (in radians)
# omega1 and omega2 are the initial angular velocities (in radians/s)
theta1 = 3.14
theta2 = 3.10
omega1 = 0.0
omega2 = 0.0

# Set the length and mass of the two pendulum rods
l1 = 1.0
l2 = 1.0
m1 = 1.0
m2 = 1.0

# Define the time points at which to solve the equations
# Here we will use a time step of 0.01 s
t = np.linspace(0, 10, 1001)


# Set up equation function for the double pendulum
def pendulum_equations(y, t, l1, l2, m1, m2):
    """Define a system of equations for the double pendulum.

    y is a vector containing theta1, theta2, omega1, and omega2
    dydt is a vector containing the derivative of these quantites
    """
    # Unpack the input vector
    theta1, theta2, omega1, omega2 = y

    # Calculate derivatives
    dtheta1dt = omega1
    dtheta2dt = omega2
    domega1dt = -g * (2 * m1 + m2) * np.sin(theta1) - \
        m2 * g * np.sin(theta1 - 2 * theta2) - \
        2 * np.sin(theta1 - theta2) * m2 * \
        (omega2**2 * l2 + omega1**2 * l1 * np.cos(theta1 - theta2))
    domega2dt = m2 * g * np.sin(theta1 - 2 * theta2) + \
        2 * np.sin(theta1 - theta2) * \
        (omega1**2 * l1 * (m1 + m2) +
         g * (m1 + m2) * np.cos(theta1) +
         omega2**2 * l2 * m2 * np.cos(theta1 - theta2))

    # Build derivative vector
    dydt = [dtheta1dt, dtheta2dt, domega1dt, domega2dt]

    return dydt


# Solve the equations of motion for the double pendulum
y = odeint(pendulum_equations, [theta1, theta2, omega1, omega2], t,
           args=(l1, l2, m1, m2))

# Unpack the solution
theta1, theta2, omega1, omega2 = y.T

# Calculate the x and y positions of the two masses
x1 = l1 * np.sin(theta1)
y1 = -l1 * np.cos(theta1)
x2 = x1 + l2 * np.sin(theta2)
y2 = y1 - l2 * np.cos(theta2)

# Create a figure
fig, ax = plt.subplots(figsize=(5, 5))

# Loop through times
for i in range(0, len(t), 1):

    # Set the plot axis
    timeString = "{:.2f} seconds".format(t[i])
    ax.set_title("Pendulum locations at "+timeString)

    # Set the limits of the plot
    ax.set_xlim(-l1 - l2 - 0.1, l1 + l2 + 0.1)
    ax.set_ylim(-l1 - l2 - 0.1, l1 + l2 + 0.1)

    # Plot the positions of the two masses and a line for the rods
    ax.plot([0, x1[i]], [0, y1[i]], color="red", linewidth=1)
    ax.plot([x1[i], x2[i]], [y1[i], y2[i]], color="blue", linewidth=1)
    ax.plot(x1[i], y1[i], 'o-', color='red', markersize=10, linewidth=3,
            label="$m_1$")
    ax.plot(x2[i], y2[i], 'o-', color='blue', markersize=10, linewidth=3,
            label="$m_2$")

    # Calculate the center of mass and add this to the plot
    # xc = (m1*x1[i]+m2*x2[i])/(m1+m2)
    # yc = (m1*y1[i]+m2*y2[i])/(m1+m2)
    # ax.plot(xc, yc, 'o-', color="green", markersize=5, linewidth=2,
    #         label="$m_c$")

    # Add the plot legend
    ax.legend()

    # Pause the plot for a moment
    plt.pause(0.001)

    # Clear the plot to update the results
    if i != len(t)-1:
        ax.clear()

# Show the plot
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


import matplotlib.animation as animation

G = 9.8  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in m
L2 = 1.0  # length of pendulum 2 in m
L = L1 + L2  # maximal length of the combined pendulum
M1 = 5.0  # mass of pendulum 1 in kg
M2 = 5.0  # mass of pendulum 2 in kg
t_stop = 10  # how many seconds to simulate
history_len = 50  # how many trajectory points to display in the trace


def derivs(t, state):
    dydx = np.zeros_like(state)

    dydx[0] = state[1]

    delta = state[2] - state[0]
    den1 = (M1+M2) * L1 - M2 * L1 * np.cos(delta) * np.cos(delta)
    dydx[1] = ((M2 * L1 * state[1] * state[1] * np.sin(delta) * np.cos(delta)
                + M2 * G * np.sin(state[2]) * np.cos(delta)
                + M2 * L2 * state[3] * state[3] * np.sin(delta)
                - (M1+M2) * G * np.sin(state[0]))
               / den1)

    dydx[2] = state[3]

    den2 = (L2/L1) * den1
    dydx[3] = ((- M2 * L2 * state[3] * state[3] * np.sin(delta) * np.cos(delta)
                + (M1+M2) * G * np.sin(state[0]) * np.cos(delta)
                - (M1+M2) * L1 * state[1] * state[1] * np.sin(delta)
                - (M1+M2) * G * np.sin(state[2]))
               / den2)

    return dydx

# create a time array from 0..t_stop sampled at 0.01 second steps
dt = 0.01
t = np.arange(0, t_stop, dt)

# th1 and th2 are the initial angles (degrees)
# w10 and w20 are the initial angular velocities (degrees per second)
th1 = 90.0
w1 = 0.0
th2 = 90.0
w2 = 0.0

# initial state
state = np.radians([th1, w1, th2, w2])

# integrate the ODE using Euler's method
y = np.empty((len(t), 4))
y[0] = state
for i in range(1, len(t)):
    y[i] = y[i - 1] + derivs(t[i - 1], y[i - 1]) * dt

# A more accurate estimate could be obtained e.g. using scipy:
#
#   y = scipy.integrate.solve_ivp(derivs, t[[0, -1]], state, t_eval=t).y.T

theta1 = y[:, 0]
theta2 = y[:, 2]

x1 = L1*np.sin(theta1)
y1 = -L1*np.cos(theta1)

x2 = L2*np.sin(theta2) + x1
y2 = -L2*np.cos(theta2) + y1

# ---------------------------------------------------------------------
# Save dataset: t, theta1, theta2, x1, y1, x2, y2
# ---------------------------------------------------------------------
data = pd.DataFrame({
    't': t,
    'theta1': theta1,
    'theta2': theta2,
    'x1': x1,
    'y1': y1,
    'x2': x2,
    'y2': y2,
})
#data.to_csv('double_pendulum_data.csv', index=False)
#print(f"Dataset saved to double_pendulum_data.csv "
#      f"({len(data)} samples at {1/dt:.0f} Hz)")

# If you want the dataset at a *different* sampling rate than the
# integration step dt, resample by slicing, e.g. every 5th row for 20 Hz
# when dt = 0.01 (100 Hz):
#
#   sample_every = int(round((1/target_fs) / dt))
#   data.iloc[::sample_every].to_csv('double_pendulum_data.csv', index=False)

# ---------------------------------------------------------------------
# Animation with a fading / limited-length trace
# ---------------------------------------------------------------------
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(autoscale_on=False, xlim=(-L-0.5, L+0.5), ylim=(-L-0.5, L+0.5))
ax.set_aspect('equal')
ax.axis('off')
ax.grid(False)

line, = ax.plot([], [], 'o-', lw=2)
trace, = ax.plot([], [], '-', lw=1, ms=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]

    # Only keep the last `history_len` points -> trace disappears
    # behind the pendulum instead of growing forever
    start = max(0, i - history_len)
    history_x = x2[start:i]
    history_y = y2[start:i]

    line.set_data(thisx, thisy)
    trace.set_data(history_x, history_y)
    time_text.set_text(time_template % (i*dt))
    return line, trace, time_text


ani = animation.FuncAnimation(
    fig, animate, len(y), interval=dt*1000, blit=True)
plt.show()

# To save the animation instead of / in addition to showing it:
# ani.save('double_pendulum.mp4', fps=int(1/dt))
# or
# ani.save('double_pendulum.gif', writer='pillow', fps=int(1/dt))
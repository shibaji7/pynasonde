import matplotlib.pyplot as plt
import numpy as np


def ionosphere_model(height, f_plasma_max, height_max):
    """Simplified ionosphere model (single layer)"""
    if height <= 100 * 1e3:
        print(">>", height / 1e3, 0)
        return 0
    if height > 1000 * 1e3:
        print(">>>", height / 1e3, 0)
        return 0
    elif height < height_max:
        # Parabolic approximation for electron density profile
        f_plasma_layer = f_plasma_max * np.sqrt(
            1 - ((height - height_max) / height_max) ** 2
        )
        print(">>>>>", height / 1e3, f_plasma_layer, f_plasma_max)
        return f_plasma_layer
    else:
        return 0


def refractive_index(f, f_plasma, f_gyro):
    """Calculates the refractive index for a collisionless medium (simplified)"""
    if f <= f_plasma:  # no propagation
        return 0
    x = (f_plasma / f) ** 2
    return np.sqrt(1 - x)  # assumes no magnetic field.


def runge_kutta4(
    initial_state,
    h,
    num_steps,
    f,
    f_plasma_max,
    height_max,
    f_gyro,
    earth_radius=6371e3,
):
    """
    Performs numerical integration using 4th order Runge-Kutta method to
    compute the ray path. This function returns a list of trajectory (x,z) coordinates.
    """
    state = initial_state
    trajectory = [state[:2]]

    for _ in range(num_steps):
        k1 = h * derivate(state, f, f_plasma_max, height_max, f_gyro, earth_radius)
        k2 = h * derivate(
            state + k1 / 2, f, f_plasma_max, height_max, f_gyro, earth_radius
        )
        k3 = h * derivate(
            state + k2 / 2, f, f_plasma_max, height_max, f_gyro, earth_radius
        )
        k4 = h * derivate(state + k3, f, f_plasma_max, height_max, f_gyro, earth_radius)

        state = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        trajectory.append(state[:2])
    return np.array(trajectory)


def derivate(state, f, f_plasma_max, height_max, f_gyro, earth_radius):
    """
    Computes the derivatives of the state variables in x and z directions for
    ray tracing. This is based on the differential equations in Haselgrove equations.
    Assumes no magnetic field or collisions.

    Input state [x, z, theta_x, theta_z]
    """
    x, z, theta_x, theta_z = state

    f_plasma = ionosphere_model(z, f_plasma_max, height_max)
    n = refractive_index(f, f_plasma, f_gyro)  # refractive index based on model

    if n <= 0:  # no propagation condition
        return np.array([0, 0, 0, 0])

    deriv_x = np.cos(theta_x)
    deriv_z = np.sin(theta_z)
    # simplified equation for non magnetic medium where k is the wavenumber, and is constant
    deriv_theta_x = 0  # for non magnetic field
    deriv_theta_z = 0  # for non magnetic field
    # for a homogeneous medium with no change in refractive index,
    # theta will not change

    return np.array([deriv_x, deriv_z, deriv_theta_x, deriv_theta_z])


def plot_ray_tracing(
    trajectory,
    f,
    f_plasma_max,
    height_max,
    f_gyro,
    earth_radius,
    azimuth,
    elevation,
):
    """Plots the ray trajectory and ionosphere profile."""

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot ray trajectory
    x_points, z_points = trajectory[:, 0], trajectory[:, 1]
    x_scaled = x_points
    z_scaled = z_points + earth_radius
    ax.plot(x_scaled / 1000, z_scaled / 1000, label="Ray Path")

    # Plot the simplified ionosphere model
    heights = np.linspace(0, 400000, 200)
    f_plasma_values = [ionosphere_model(h, f_plasma_max, height_max) for h in heights]
    z_values = (heights + earth_radius) / 1000
    plt.plot(
        [0] * len(z_values), z_values, linestyle="--", label="Ionosphere", c="grey"
    )  # x axis location arbitrary.
    ax.axhline(
        y=(height_max + earth_radius) / 1000, color="r", linestyle=":", label="hmax"
    )

    ax.set_xlabel("Ground Range (km)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title(
        f"Simplified 2D Ray Tracing for f={f/1e6} MHz, elevation:{elevation} deg, Azimuth: {azimuth} deg"
    )
    ax.grid(True)
    ax.legend()
    plt.savefig("tmp/00_rt.png")


if __name__ == "__main__":
    # --- Simulation Parameters ---
    f = 3e6  # Frequency (Hz)
    f_plasma_max = 10e6  # Maximum plasma frequency (Hz)
    height_max = 300000  # Height of the layer (m)
    f_gyro = 1.2e6  # gyrofrequency at 0 degree latitude in (Hz).
    earth_radius = 6371e3  # Earth radius (m)

    azimuth = 0  # angle with respect to x axis
    elevation = np.deg2rad(70)  # angle from the ground (10 degrees)

    # -- Numerical Integration Settings --
    num_steps = 10000  # Number of integration steps
    h = 1000  # Step size (m), 1km step size

    # -- Initial State --
    initial_state = np.array(
        [0, 0, np.pi / 2 - elevation, elevation]
    )  # [x, z, theta_x, theta_z] Initial position, angle, and z velocity.

    # --- Perform Ray Tracing Simulation ---
    trajectory = runge_kutta4(
        initial_state, h, num_steps, f, f_plasma_max, height_max, f_gyro, earth_radius
    )
    # -- Plotting the results
    plot_ray_tracing(
        trajectory,
        f,
        f_plasma_max,
        height_max,
        f_gyro,
        earth_radius,
        azimuth,
        np.rad2deg(elevation),
    )

import math
from typing import Any, Dict, List

from rlgym.api import TransitionEngine
from rlgym.cartpole.api import Cart, CartPoleConfig, CartPoleState, Pole


class DefaultEngine(TransitionEngine[str, CartPoleState, int]):
    CONFIG_KEY_GRAVITY = "gravity"
    CONFIG_KEY_FORCE_MAGNITUDE = "force_magnitude"
    CONFIG_KEY_TAU = "seconds_between_update"
    CONFIG_KEY_KINEMATICS = "kinematics_integrator"
    CONFIG_X_THRESHOLD = "x_threshold"
    CONFIG_THETA_THRESHOLD = "theta_threshold"

    def __init__(self) -> None:
        self.gravity = 9.8
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 20.4
        self._config = {
            self.CONFIG_KEY_GRAVITY: self.gravity,
            self.CONFIG_KEY_FORCE_MAGNITUDE: self.force_mag,
            self.CONFIG_KEY_TAU: self.tau,
            self.CONFIG_KEY_KINEMATICS: "euler",
            self.CONFIG_THETA_THRESHOLD: self.theta_threshold_radians,
            self.CONFIG_X_THRESHOLD: self.x_threshold,
        }

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    @config.setter
    def config(self, value: Dict[str, Any]):
        for key in [
            self.CONFIG_KEY_GRAVITY,
            self.CONFIG_KEY_FORCE_MAGNITUDE,
            self.CONFIG_KEY_TAU,
            self.CONFIG_KEY_KINEMATICS,
            self.CONFIG_X_THRESHOLD,
            self.CONFIG_THETA_THRESHOLD,
        ]:
            if key in value:
                self._config[key] = value[key]

    @property
    def max_num_agents(self) -> int:
        return 1

    @property
    def state(self) -> CartPoleState:
        return self._state

    @property
    def agents(self) -> List[str]:
        return ["cart"]

    def create_base_state(self) -> CartPoleState:
        self._state = CartPoleState(
            CartPoleConfig(self.x_threshold, self.theta_threshold_radians),
            Pole(0.1, 0.5, 0, 0),
            Cart(1.0, 0, 0),
        )
        return self._state

    def step(
        self, actions: Dict[str, int], shared_info: Dict[str, Any]
    ) -> CartPoleState:
        x, x_dot, theta, theta_dot = (
            self._state.cart.x,
            self._state.cart.x_dot,
            self._state.pole.theta,
            self._state.pole.theta_dot,
        )
        force = self.force_mag if actions["cart"] == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        _pole_masslength = self._state.pole.mass * self._state.pole.length
        _total_mass = self._state.cart.mass + self._state.pole.mass

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + _pole_masslength * pow(theta_dot, 2) * sintheta) / _total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self._state.pole.length
            * (4.0 / 3.0 - self._state.pole.mass * pow(costheta, 2) / _total_mass)
        )
        xacc = temp - _pole_masslength * thetaacc * costheta / _total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self._state.cart.x = x
        self._state.cart.x_dot = x_dot
        self._state.pole.theta = theta
        self._state.pole.theta_dot = theta_dot

        return self._state

    def set_state(
        self, desired_state: CartPoleState, shared_info: Dict[str, Any]
    ) -> CartPoleState:

        self.x_threshold = desired_state.config.x_threshold
        self.theta_threshold_radians = desired_state.config.theta_threshold

        _new_x = max(min(desired_state.cart.x, self.x_threshold), -self.x_threshold)
        _new_theta = max(
            min(desired_state.pole.theta, self.theta_threshold_radians),
            -self.theta_threshold_radians,
        )

        self._state.cart.x = _new_x
        self._state.cart.x_dot = desired_state.cart.x_dot
        self._state.cart.mass = max(desired_state.cart.mass, 1e-8)

        self._state.pole.mass = max(desired_state.pole.mass, 1e-8)
        self._state.pole.length = max(desired_state.pole.length, 0.1)
        self._state.pole.theta = _new_theta
        self._state.pole.theta_dot = desired_state.pole.theta_dot

        self._state.config.theta_threshold = self.theta_threshold_radians

        return self._state

    def close(self) -> None:
        pass

from dataclasses import dataclass


@dataclass
class CartPoleConfig:
    """The basic config for a cartpole enviroment"""

    x_threshold: float
    theta_threshold: float


@dataclass
class Object:
    """A base class containing a mass for all objects (cart and pole)"""

    mass: float


@dataclass
class Cart(Object):
    """This represents the cart in the environment"""

    x: float
    x_dot: float


@dataclass
class Pole(Object):
    """This represents the pole in the environment"""

    length: float
    theta: float
    theta_dot: float


@dataclass
class CartPoleState:
    """The state contains a pole, a cart and a config"""

    config: CartPoleConfig
    pole: Pole
    cart: Cart

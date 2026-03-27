from dataclasses import dataclass, field

import Box2D


@dataclass
class LunarLanderConfig:
    gravity: float = -10
    main_engine_power: float = 13.0
    lateral_engine_power: float = 0.6
    wind_power: float = 0
    turbulence_power: float = 0


@dataclass(init=False)
class LunarLanderState:
    """
    Holds the mutable simulation state shared across all components.

    Analogous to the GameState object in RLGym.  All components receive a
    reference to the same instance so they can read/write physics objects
    without coupling to one another.
    """

    world: Box2D.b2World
    moon: Box2D.b2Body
    lander: Box2D.b2Body
    legs: list[Box2D.b2Body]
    particles: list[Box2D.b2Body] = field(default_factory=lambda: [])
    sky_polys: list[list[tuple[float, float]]]
    helipad_x1: float
    helipad_x2: float
    helipad_y: float
    game_over: bool
    config: LunarLanderConfig

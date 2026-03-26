import math
from typing import Any, Dict, List

from Box2D import (
    b2EdgeShape,
    b2FixtureDef,
    b2CircleShape,
    b2ContactListener,
    b2Contact,
    b2PolygonShape,
    b2RevoluteJointDef,
    b2World,
)
import numpy as np

from rlgym.api import TransitionEngine

from rlgym.lunar_lander.common_values import (
    AGENT_NAME,
    TICKS_PER_SECOND,
    LANDER_POLY,
    LEG_AWAY,
    LEG_DOWN,
    LEG_H,
    LEG_SPRING_TORQUE,
    LEG_W,
    MAIN_ENGINE_Y_LOCATION,
    SCALE,
    SIDE_ENGINE_AWAY,
    SIDE_ENGINE_HEIGHT,
    VIEWPORT_H,
    VIEWPORT_W,
)
from rlgym.lunar_lander.state import LunarLanderConfig, LunarLanderState


class _ContactDetector(b2ContactListener):
    def __init__(self, state: LunarLanderState) -> None:
        b2ContactListener.__init__(self)
        self._state = state

    def BeginContact(self, contact: b2Contact) -> None:
        s = self._state
        if s.lander in (contact.fixtureA.body, contact.fixtureB.body):
            s.game_over = True
        for i in range(2):
            if s.legs[i] in (contact.fixtureA.body, contact.fixtureB.body):
                print(f"Contact {i}")
                s.legs[i].ground_contact = True

    def EndContact(self, contact) -> None:
        s = self._state
        for i in range(2):
            if s.legs[i] in (contact.fixtureA.body, contact.fixtureB.body):
                s.legs[i].ground_contact = False


class LunarLanderTransitionEngine(TransitionEngine[str, LunarLanderState, np.ndarray]):
    """
    Advances the Box2D simulation by one timestep and returns an updated
    state (with engine forces applied and the world stepped).

    Corresponds to TransitionEngine in the RLGym API.
    """

    def __init__(self, seed: int = 123) -> None:
        self._state = None
        self._rng = np.random.RandomState(seed)

    @property
    def config(self) -> Dict[str, Any]:
        return {}

    @config.setter
    def config(self, value: dict[str, Any]):
        pass

    @property
    def max_num_agents(self) -> int:
        return 1

    @property
    def agents(self) -> List[str]:
        return [AGENT_NAME]

    @property
    def state(self) -> LunarLanderState:
        assert self._state is not None
        return self._state

    def create_base_state(self) -> LunarLanderState:
        self._state = LunarLanderState()

        self._state.config = LunarLanderConfig()

        self._state.world = b2World(gravity=(0, self._state.config.gravity))
        detector = _ContactDetector(self._state)
        self._state.world.contactListener = detector

        self._state.game_over = False

        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        # --- Terrain ---
        CHUNKS = 11
        height = self._rng.uniform(0, H / 2, size=(CHUNKS + 1,))
        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]

        self._state.helipad_x1 = chunk_x[CHUNKS // 2 - 1]
        self._state.helipad_x2 = chunk_x[CHUNKS // 2 + 1]
        self._state.helipad_y = H / 4

        for idx in range(-2, 3):
            height[CHUNKS // 2 + idx] = self._state.helipad_y

        smooth_y = [
            0.33 * (height[i - 1] + height[i] + height[i + 1]) for i in range(CHUNKS)
        ]

        self._state.moon = self._state.world.CreateStaticBody(
            shapes=b2EdgeShape(vertices=[(0, 0), (W, 0)])
        )

        self._state.sky_polys = []
        for i in range(CHUNKS - 1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i + 1], smooth_y[i + 1])
            self._state.moon.CreateEdgeFixture(
                vertices=[p1, p2], density=0, friction=0.1
            )
            self._state.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])
        self._state.moon.color1 = (0.0, 0.0, 0.0)
        self._state.moon.color2 = (0.0, 0.0, 0.0)

        # --- Lander body ---
        initial_x = VIEWPORT_W / SCALE / 2
        initial_y = VIEWPORT_H / SCALE
        self._state.lander = self._state.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=b2FixtureDef(
                shape=b2PolygonShape(
                    vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]
                ),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,
                restitution=0.0,
            ),
        )
        self._state.lander.color1 = (128, 102, 230)
        self._state.lander.color2 = (77, 77, 128)

        # --- Legs ---
        self._state.legs = []
        for i in [-1, +1]:
            leg = self._state.world.CreateDynamicBody(
                position=(initial_x - i * LEG_AWAY / SCALE, initial_y),
                angle=(i * 0.05),
                fixtures=b2FixtureDef(
                    shape=b2PolygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001,
                ),
            )
            leg.ground_contact = False
            leg.color1 = (128, 102, 230)
            leg.color2 = (77, 77, 128)

            rjd = b2RevoluteJointDef(
                bodyA=self._state.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3 * i,
            )
            if i == -1:
                rjd.lowerAngle = +0.9 - 0.5
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self._state.world.CreateJoint(rjd)
            self._state.legs.append(leg)

        self._state.particles = []

        return self._state

    # ------------------------------------------------------------------

    def _destroy(self) -> None:
        if self._state is None or self._state.world is None:
            return
        self._state.world.contactListener = None
        # Clean all particles
        for p in self._state.particles:
            self._state.world.DestroyBody(p)
        self._state.particles.clear()
        if self._state.moon:
            self._state.world.DestroyBody(self._state.moon)
            self._state.moon = None
        if self._state.lander:
            self._state.world.DestroyBody(self._state.lander)
            self._state.lander = None
        for leg in self._state.legs:
            self._state.world.DestroyBody(leg)
        self._state.legs = []

    def step(
        self, actions: Dict[str, np.ndarray], shared_info: Dict[str, Any]
    ) -> LunarLanderState:
        action = actions[AGENT_NAME]

        # --- Wind ---
        if not (
            self._state.legs[0].ground_contact or self._state.legs[1].ground_contact
        ):
            wi = shared_info.get("wind_idx", 0)
            wind_mag = (
                math.tanh(math.sin(0.02 * wi) + math.sin(math.pi * 0.01 * wi))
                * self.state.config.wind_power
            )
            shared_info["wind_idx"] = wi + 1
            self.state.lander.ApplyForceToCenter((wind_mag, 0.0), True)

            ti = shared_info.get("torque_idx", 0)
            torque_mag = (
                math.tanh(math.sin(0.02 * ti) + math.sin(math.pi * 0.01 * ti))
                * self.state.config.turbulence_power
            )
            shared_info["torque_idx"] = ti + 1
            self.state.lander.ApplyTorque(torque_mag, True)

        # --- Engine impulses ---
        tip = (math.sin(self.state.lander.angle), math.cos(self.state.lander.angle))
        side = (-tip[1], tip[0])
        dispersion = [np.random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        m_power = 0.0
        if action[0] > 0.0:
            m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5

            ox = (
                tip[0] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                + side[0] * dispersion[1]
            )
            oy = (
                -tip[1] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                - side[1] * dispersion[1]
            )
            impulse_pos = (
                self.state.lander.position[0] + ox,
                self.state.lander.position[1] + oy,
            )
            p = self._create_particle(
                self.state, 3.5, impulse_pos[0], impulse_pos[1], m_power
            )
            p.ApplyLinearImpulse(
                (
                    ox * self.state.config.main_engine_power * m_power,
                    oy * self.state.config.main_engine_power * m_power,
                ),
                impulse_pos,
                True,
            )
            self.state.lander.ApplyLinearImpulse(
                (
                    -ox * self.state.config.main_engine_power * m_power,
                    -oy * self.state.config.main_engine_power * m_power,
                ),
                impulse_pos,
                True,
            )

        s_power = 0.0
        if np.abs(action[1]) > 0.5:
            direction = np.sign(action[1])
            s_power = np.clip(np.abs(action[1]), 0.5, 1.0)

            ox = tip[0] * dispersion[0] + side[0] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )
            oy = -tip[1] * dispersion[0] - side[1] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )
            impulse_pos = (
                self.state.lander.position[0] + ox - tip[0] * 17 / SCALE,
                self.state.lander.position[1]
                + oy
                + tip[1] * SIDE_ENGINE_HEIGHT / SCALE,
            )
            p = self._create_particle(
                self.state, 0.7, impulse_pos[0], impulse_pos[1], s_power
            )
            p.ApplyLinearImpulse(
                (
                    ox * self.state.config.lateral_engine_power * s_power,
                    oy * self.state.config.lateral_engine_power * s_power,
                ),
                impulse_pos,
                True,
            )
            self.state.lander.ApplyLinearImpulse(
                (
                    -ox * self.state.config.lateral_engine_power * s_power,
                    -oy * self.state.config.lateral_engine_power * s_power,
                ),
                impulse_pos,
                True,
            )

        self.state.world.Step(1.0 / TICKS_PER_SECOND, 6 * 30, 2 * 30)

        # Store engine powers for reward function
        shared_info["m_power"] = m_power
        shared_info["s_power"] = s_power

        return self.state

    def set_state(
        self, desired_state: LunarLanderState, shared_info: Dict[str, Any]
    ) -> LunarLanderState:
        self.state.config = desired_state.config
        return self.state

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    @staticmethod
    def _create_particle(
        state: LunarLanderState, mass: float, x: float, y: float, ttl: float
    ):
        p = state.world.CreateDynamicBody(
            position=(x, y),
            angle=0.0,
            fixtures=b2FixtureDef(
                shape=b2CircleShape(radius=2 / SCALE, pos=(0, 0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001,
                restitution=0.3,
            ),
        )
        p.ttl = ttl
        state.particles.append(p)
        # Clean expired particles
        while state.particles and state.particles[0].ttl < 0:
            state.world.DestroyBody(state.particles.pop(0))
        return p

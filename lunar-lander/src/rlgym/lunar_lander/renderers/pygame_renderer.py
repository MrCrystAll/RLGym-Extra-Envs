from Box2D import b2CircleShape

try:
    import pygame
    from pygame import gfxdraw
except ImportError as exc:
    raise ImportError(
        'pygame is not installed. Run "pip install rlgym-lunar-lander[render]" to install it.'
    ) from exc

from rlgym.api import Renderer

from rlgym.lunar_lander.api.common_values import (
    TICKS_PER_SECOND,
    SCALE,
    VIEWPORT_H,
    VIEWPORT_W,
)
from rlgym.lunar_lander.api.state import LunarLanderState


class PygameRenderer(Renderer[LunarLanderState]):
    """
    Renders the simulation using pygame.

    Corresponds to Renderer in the RLGym API.
    """

    def __init__(self) -> None:
        self._screen = None
        self._clock = None
        self._surf = None

    def render(self, state: LunarLanderState, shared_info: dict):
        if self._screen is None:
            pygame.init()
            pygame.display.init()
            self._screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        if self._clock is None:
            self._clock = pygame.time.Clock()

        surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))
        pygame.draw.rect(surf, (255, 255, 255), surf.get_rect())

        for obj in state.particles:
            obj.ttl -= 0.15
            c = (
                int(max(0.2, 0.15 + obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
            )
            obj.color1 = obj.color2 = c
        while state.particles and state.particles[0].ttl < 0:
            state.world.DestroyBody(state.particles.pop(0))

        for p in state.sky_polys:
            scaled = [(c[0] * SCALE, c[1] * SCALE) for c in p]
            pygame.draw.polygon(surf, (0, 0, 0), scaled)
            gfxdraw.aapolygon(surf, scaled, (0, 0, 0))

        drawlist = [state.lander] + state.legs
        for obj in state.particles + drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if isinstance(f.shape, b2CircleShape):
                    pygame.draw.circle(
                        surf,
                        obj.color1,
                        trans * f.shape.pos * SCALE,
                        f.shape.radius * SCALE,
                    )
                    pygame.draw.circle(
                        surf,
                        obj.color2,
                        trans * f.shape.pos * SCALE,
                        f.shape.radius * SCALE,
                    )
                else:
                    path = [trans * v * SCALE for v in f.shape.vertices]
                    pygame.draw.polygon(surf, obj.color1, path)
                    gfxdraw.aapolygon(surf, path, obj.color1)
                    pygame.draw.aalines(surf, obj.color2, True, path)

        for x in [state.helipad_x1, state.helipad_x2]:
            x = x * SCALE
            flagy1 = state.helipad_y * SCALE
            flagy2 = flagy1 + 50
            pygame.draw.line(surf, (255, 255, 255), (x, flagy1), (x, flagy2), 1)
            pygame.draw.polygon(
                surf,
                (204, 204, 0),
                [(x, flagy2), (x, flagy2 - 10), (x + 25, flagy2 - 5)],
            )

        surf = pygame.transform.flip(surf, False, True)

        self._screen.blit(surf, (0, 0))
        pygame.event.pump()
        self._clock.tick(TICKS_PER_SECOND)
        pygame.display.flip()

    def close(self) -> None:
        if self._screen is not None:
            pygame.display.quit()
            pygame.quit()
            self._screen = None

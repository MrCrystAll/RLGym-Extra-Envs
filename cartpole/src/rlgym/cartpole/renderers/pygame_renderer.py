from typing import Any, Dict

from rlgym.api import Renderer

try:
    import pygame
    from pygame import gfxdraw
except ImportError as exc:
    raise ImportError(
        'pygame is not installed. Run "pip install rlgym-cartpole[render]" to install it.'
    ) from exc

from rlgym.cartpole.api.state import CartPoleState


class PygameRenderer(Renderer[CartPoleState]):
    def __init__(
        self, screen_width: int = 600, screen_height: int = 400, fps_render: int = 60
    ) -> None:
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.fps_render = fps_render
        self.surf = pygame.Surface((self.screen_width, self.screen_height))

        self.screen = None
        self.clock = None

    def render(self, state: CartPoleState, shared_info: Dict[str, Any]) -> Any:
        if self.screen is None:
            pygame.display.init()
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
            )

        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = state.config.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * state.pole.length)
        cartwidth = 50.0
        cartheight = 30.0

        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = state.cart.x * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-state.pole.theta)
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        pygame.event.pump()
        self.clock.tick(self.fps_render)
        pygame.display.flip()

    def close(self):
        pass

import numpy as np
from dm_control.composer.arena import Arena
from arenas.hills import SineBumps
from dm_control.composer import entity
from dm_control import mjcf 
from dm_control.composer.observation import observable

class ScentSource(entity.Entity):
    """A spherical scent source with linear decay."""
    
    def _build(self, radius=0.5, max_concentration=1.0, max_range=5.0):
        self._radius = radius
        self._max_concentration = max_concentration
        self._max_range = max_range
        
        # Create visual geometry for the scent source
        self._mjcf_root = mjcf.RootElement()
        self._mjcf_root.worldbody.add(
            'geom',
            name='scent_source',
            type='sphere',
            size=[radius],
            rgba=[1, 0, 0, 1],  # Red color
            contype=0,  # No collision
            conaffinity=0)
    
    def get_concentration(self, position):
        """Calculate scent concentration at given position."""
        dist = np.linalg.norm(position - self.pos)
        if dist > self._max_range:
            return 0.0
        return max(0.0, self._max_concentration * (1 - dist / self._max_range))
    
    @property
    def pos(self):
        return self._mjcf_root.find('geom', 'scent_source').pos
    
    @pos.setter
    def pos(self, value):
        self._mjcf_root.find('geom', 'scent_source').pos = value

class SineBumpsWithScent(SineBumps):
    """SineBumps arena with added scent source."""
    
    def _build(self,
               name='sine_bumps_with_scent',
               scent_radius=0.5,
               scent_max_concentration=1.0,
               scent_max_range=5.0,
               **kwargs):
        super()._build(name=name, **kwargs)
        
        # Add scent source
        self._scent_source = ScentSource(
            radius=scent_radius,
            max_concentration=scent_max_concentration,
            max_range=scent_max_range)
        self.attach(self._scent_source)
        
        # Initialize scent source position (can be randomized in initialize_episode)
        self._scent_source.pos = [0, 0, 2]  # Example position
    
    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        
        # Randomize scent source position if desired
        x = random_state.uniform(-self._dim + 2, self._dim - 2)
        y = random_state.uniform(-self._dim + 2, self._dim - 2)
        z = 2.0  # Height above ground
        self._scent_source.pos = [x, y, z]
    
    def get_scent_concentration(self, physics, walker_pos):
        """Get scent concentration at walker's position."""
        return self._scent_source.get_concentration(walker_pos)
    
    def add_observables(self, walker):
        """Add scent concentration observable to the walker."""
        super().add_observables(walker)
        
        def scent_concentration(physics):
            walker_pos = physics.bind(walker.root_body).xpos
            return np.array([self.get_scent_concentration(physics, walker_pos)])
        
        walker.observables.scent_concentration = observable.Generic(scent_concentration)
        walker.observables.scent_concentration.enabled = True
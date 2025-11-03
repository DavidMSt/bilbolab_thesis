import dataclasses

import numpy as np

from core.utils.colors import NamedColor, Colors


@dataclasses.dataclass
class SimulatedAgentDefinition:
    id: str
    color: list | tuple
    fov: float = np.deg2rad(100)
    size = 0.2
    vision_radius: float = 1.5


SIMULATED_AGENTS = [
    SimulatedAgentDefinition("vfrodo1", Colors.lime),
    SimulatedAgentDefinition("vfrodo2", Colors.aquamarine),
    SimulatedAgentDefinition("vfrodo3", Colors.orange),
    SimulatedAgentDefinition("vfrodo4", Colors.coral),
    SimulatedAgentDefinition("vfrodo5", Colors.lightpurple),
]


def get_simulated_agent_definition_by_id(frodo_id: str) -> SimulatedAgentDefinition | None:
    for agent in SIMULATED_AGENTS:
        if agent.id == frodo_id:
            return agent
    return None

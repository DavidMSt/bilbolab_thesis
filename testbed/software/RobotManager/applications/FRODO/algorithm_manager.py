import dataclasses

from applications.FRODO.algorithm.algorithm_centralized import CentralizedAlgorithm
from applications.FRODO.algorithm.algorithm_distributed import DistributedAlgorithm

"""
The algorithm manager manages the running algorithms, sets them up with parameters and updates them based on 
given agent measurements and inputs
"""


@dataclasses.dataclass
class AlgorithmAgentContainer:
    id: str


class FRODO_AlgorithmManager:
    algorithm_centralized: CentralizedAlgorithm
    algorithm_distributed: DistributedAlgorithm

    # === INIT =========================================================================================================
    def __init__(self):
        ...

    # === METHODS ======================================================================================================
    def initialize(self):
        ...

    # ------------------------------------------------------------------------------------------------------------------
    def update(self):
        ...

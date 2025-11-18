from general.general_simulation import FRODO_general_Simulation, FrodoGeneralEnvironment
from universal.universal_agent import UniversalAgent
from motion_planning.mp_simulation import MPSimulationModule
from task_assignment.task_simulation import AssignmentSimulationModule

class UniversalSimulation(FRODO_general_Simulation):
    def __init__(self, Ts=0.1, use_web_interface: bool = False, limits: tuple[tuple[int, int], ...] = ..., env=...):
        super().__init__(Ts, use_web_interface, limits, env= env)
        self.mpi = MPSimulationModule
        self.asi = AssignmentSimulationModule

if __name__ == "__main__":
    ...

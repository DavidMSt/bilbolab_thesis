from extensions.cli.src.cli import Command, CommandSet
from robots.frodo.archive.frodo_manager import FrodoManager


class FrodoManager_Commands(CommandSet):
    manager: FrodoManager

    def __init__(self, manager: FrodoManager):
        self.manager = manager

        stop_command = Command("stop",
                               callback=self.stop_all,
                               arguments=[],
                               description="Stops all robots")

        list_command = Command("list",
                               callback=self.list_agents,
                               arguments=[],
                               description="List all agents")

        super().__init__(name='robots', commands=[stop_command, list_command], child_sets=[])

    def stop_all(self):
        for id, robot in self.manager.robots.items():
            robot.setSpeed(0, 0)

        return "Stop all robots"

    def list_agents(self):
        if len(self.manager.robots) == 0:
            return "No robots"

        out = []

        for id, robot in self.manager.robots.items():
            out.append(id)

        return "Robots: " + ", ".join(out)

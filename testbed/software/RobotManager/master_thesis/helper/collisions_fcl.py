import numpy as np
import yaml
import fcl
import argparse
# import meshcat.transformations as tf

class CollisionChecker():
    # _env: dict # environment in which agent is
    # _plan: dict # plan of configurations we want to check for collisions

    def __init__(self, planning_config, type = None):
        self._env = None  # Ensure _env is always defined
        self.collisions_list = [] # initialize dynamic collison list
        self._planning_config = planning_config
        self._type = type

    def run_from_yaml(self):
        # load data from yaml
        self.args = self.parse_arguments()
        # extract env and configuration plan
        env, plan = self.load_data()
        # initialize the environment
        self.initialize_env(env)
        # set configuration plan to be checke
        self.set_plan(plan)
        # run the collision checking for each state of the plan
        self.broadphase_collision_checking(self.states)

    def set_plan(self, plan):
        # self._plan = plan
        self._type = plan["plan"]["type"]
        self.states = plan["plan"]["states"]
        if self._type == "arm":
            self.set_dimensions(plan["plan"]["L"])

        elif self._type == "car":
            self.set_dimensions(plan["plan"]["L"], plan["plan"]["W"], plan["plan"]["H"])
    
    def set_dimensions(self, L, W= None, H= None):
        if self._type == 'arm':
            self.dimensions = {
                "L": L
            }
        elif self._type == 'car':
            self.dimensions = {
                "L": L,
                "W": W,
                "H": H
            }
        elif self._type == "frodo":
            self.dimensions = {
                "L": L,
                "W": W,
                "H": H
            }
        self.dimensions

    def initialize_env(self, env= None):
        self._env = env
        # Defensive programming: check for 'obstacles' attribute
        if not hasattr(env, "obstacles"):
            raise AttributeError("Provided env has no 'obstacles' attribute")
        obstacle_list = self.create_environment_objects()
        self.env_manager = self.create_collision_manager(obstacle_list)

    @staticmethod
    def parse_arguments():
        parser = argparse.ArgumentParser()
        parser.add_argument('env', help='input YAML file with environment')
        parser.add_argument('plan', help='input YAML file with plan')
        parser.add_argument('output', help='output yaml-file')
        args = parser.parse_args()
        return args
    
    def load_data(self):
        # load environment
        with open(self.args.env, "r") as stream:
            env = yaml.safe_load(stream)
        # load plan
        with open(self.args.plan, "r") as stream:
            plan = yaml.safe_load(stream)
        return env, plan
    
    def check_state(self, state):
        """Check a single configuration for collision"""
        req = fcl.CollisionRequest(num_max_contacts=1, enable_contact=False)
        data = fcl.CollisionData(request=req)
        agent_objs = self.create_agent_objects(state)
        agent_manager = self.create_collision_manager(agent_objs)
        agent_manager.collide(self.env_manager, data, fcl.defaultCollisionCallback)
        return data.result.is_collision
    
    def check_state_ompl(self, state):
        if self._type == "frodo":
            x = state.getX()
            y = state.getY()
            theta = state.getYaw()
            config = [x, y, theta]
        # elif self._type == "arm":
        #     # extract joint angles from flat real-vector state
        #     config = [float(state[i]) for i in range(len(self.dimensions["L"]))]
        else:
            raise NotImplementedError("Collision checker can't handle type: {}".format(self._type))

        valid = self.check_state(config)

        return not valid # ompl expects collision as return here

    def check_states(self, states):
        """Check multiple configurations"""
        return [self.check_state(s) for s in states]

    def broadphase_collision_checking(self, states):
        self.collisions_list = [self.check_state(state) for state in states]

    def create_agent_objects(self, state):
        if self._type == "frodo":
            objs = self.create_objects_frodo(state)

        elif self._type == "arm":
            objs = self.create_objects_arm(state)

        else:
            raise NotImplementedError(f"Unknown plan type: {self._plan['plan']['type']}")

        return objs

    def create_environment_objects(self):
        obstacle_list = []
        for obstacle in self._planning_config.obstacles:
            _obs = self.create_env_collision_object(obstacle)
            obstacle_list.append(_obs)
        return obstacle_list
    
    def create_env_collision_object(self, obstacle):
        if obstacle["type"] == "box":
            _obs = self.create_collision_box(obstacle["pos"], obstacle["size"])
        elif obstacle["type"] == "cylinder":
            _obs = self.create_collision_cylinder(obstacle["pos"], obstacle["q"], obstacle["r"], obstacle["lz"])
        else:
            raise ValueError
        return _obs
    
    # def create_objects_arm(self, state):
    #     L = self.dimensions["L"]
    #     radius = 0.04
    #     theta1, theta2, theta3 = state
    #     offset = np.pi / 2

    #     links = []
    #     transforms = []

    #     # Link 1
    #     l1_x = L[0]/2 * np.cos(theta1)
    #     l1_y = L[0]/2 * np.sin(theta1)
    #     T1 = tf.translation_matrix([l1_x, l1_y, 0]) @ tf.euler_matrix(np.pi/2, 0, offset + theta1)
    #     tf1 = fcl.Transform(T1[:3, :3], T1[:3, 3])
    #     links.append(fcl.Cylinder(radius, L[0]))
    #     transforms.append(tf1)

    #     # Link 2
    #     l2_x = L[0] * np.cos(theta1) + L[1]/2 * np.cos(theta1 + theta2)
    #     l2_y = L[0] * np.sin(theta1) + L[1]/2 * np.sin(theta1 + theta2)
    #     T2 = tf.translation_matrix([l2_x, l2_y, 0]) @ tf.euler_matrix(np.pi/2, 0, offset + theta1 + theta2)
    #     tf2 = fcl.Transform(T2[:3, :3], T2[:3, 3])
    #     links.append(fcl.Cylinder(radius, L[1]))
    #     transforms.append(tf2)

    #     # Link 3
    #     l3_x = L[0] * np.cos(theta1) + L[1] * np.cos(theta1 + theta2) + L[2]/2 * np.cos(theta1 + theta2 + theta3)
    #     l3_y = L[0] * np.sin(theta1) + L[1] * np.sin(theta1 + theta2) + L[2]/2 * np.sin(theta1 + theta2 + theta3)
    #     T3 = tf.translation_matrix([l3_x, l3_y, 0]) @ tf.euler_matrix(np.pi/2, 0, offset + theta1 + theta2 + theta3)
    #     tf3 = fcl.Transform(T3[:3, :3], T3[:3, 3])
    #     links.append(fcl.Cylinder(radius, L[2]))
    #     transforms.append(tf3)

    #     objs = [fcl.CollisionObject(geom, transform) for geom, transform in zip(links, transforms)]
    #     return objs

    def create_objects_frodo(self, state):
        x, y, theta = state

        # Use .get to avoid KeyError if values are missing
        L = self.dimensions.get("L")
        W = self.dimensions.get("W")
        H = self.dimensions.get("H")

        pos = [x, y, 0]
        q = [np.cos(theta / 2), 0, 0, np.sin(theta / 2)]  # Z-rotation quaternion

        obj = self.create_collision_box(pos, [L, W, H], q)
        objs = [obj]
        return objs

    @staticmethod
    def create_collision_box(pos, size, q = None):
        geometry = fcl.Box(*size)
        if q:
            transformation = fcl.Transform(q, pos)
        else: 
            transformation = fcl.Transform(pos)
        obj = fcl.CollisionObject(geometry, transformation)
        return obj

    @staticmethod
    def create_collision_cylinder(pos, q, radius, lz):
        geometry = fcl.Cylinder(radius, lz)
        transformation = fcl.Transform(q, pos)
        obj = fcl.CollisionObject(geometry, transformation)
        return obj

    @staticmethod
    def create_collision_manager(objects: list):
        manager = fcl.DynamicAABBTreeCollisionManager()
        manager.registerObjects(objects)
        manager.setup()
        return manager

    def dump_collisions_to_yaml(self):
        """Dump the plan to a yaml file with the given target path"""
        print(self.args.output)

        collision_content = {
            'collisions': self.collisions_list
        }

        with open(self.args.output, 'w') as file:
            yaml.dump(collision_content, file, default_flow_style=False)


if __name__ == "__main__":
    collision_checker = CollisionChecker()
    collision_checker.run_from_yaml()
    collision_checker.dump_collisions_to_yaml()

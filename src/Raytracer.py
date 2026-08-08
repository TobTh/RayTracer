import numpy as np

import CoordinateTransformations as ct


class RayTracerScene:
    def __init__(self, objects):
        self.objects = objects
        self.coordinate_systems = ct.CoordinateTransformGraph()
        self.object_names = [obj.get_name() for obj in objects];

        for obj in objects:
            self.coordinate_systems.add_transformation(obj.get_coordinate_system())

    def add_object(self, obj):
        self.objects.append(obj)
        self.object_names.append(obj.get_name())
        self.coordinate_systems.add_transformation(obj.get_coordinate_system())

    def get_object_by_name(self, name):
        for obj in self.objects:
            if obj.get_name() == name:
                return obj
        raise ValueError(f"Object with name '{name}' not found in the scene.")

    def update_object(self, name, updated_object):
        obj = self.get_object_by_name(name)
        self.coordinate_systems.add_transformation(updated_object.get_coordinate_system())

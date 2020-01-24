# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial.transform import Rotation

def build_transformation(spec, length_unit=1.0, angle_unit=1.0):
    result = np.identity(4)
    for operation, value in zip(*spec):
        translation = np.zeros(3)
        rotation = np.identity(3)
        if operation == 'TranslateX':
            translation = np.array((value*length_unit, 0, 0))
        if operation == 'TranslateY':
            translation = np.array((0, value*length_unit, 0))
        if operation == 'TranslateZ':
            translation = np.array((0, 0, value*length_unit))
        if operation == 'RotateX':
            rotation = Rotation.from_euler('x', value*angle_unit).as_dcm()
        if operation == 'RotateY':
            rotation = Rotation.from_euler('y', value*angle_unit).as_dcm()
        if operation == 'RotateZ':
            rotation = Rotation.from_euler('z', value*angle_unit).as_dcm()
        M = np.identity(4)
        M[:3,:3] = rotation
        M[:3,3] = translation
        result = M @ result
    return result

def transform(M, x):
    return (M[:3,:3] @ x) + M[:3,3]

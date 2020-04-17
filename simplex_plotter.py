import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import linear_program
from utils import zeros, array

class Simplex3DPlotter(object):
    def __init__(self, linear_program, planes, colors, scale):
        self._linear_program = linear_program
        self._previous_point = None

        assert scale.shape == (3, 2), 'scale must be a 3x2 matrix'
        assert len(planes) == len(colors), 'planes and colors must be of same length'

        fig = plt.figure()
        ax = Axes3D(fig)
        axes_lim_setters = [ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d]
        for lim_setter, (left, right) in zip(axes_lim_setters, scale):
            lim_setter(left, right)
        
        axes_label_setters = [ax.set_xlabel, ax.set_ylabel, ax.set_zlabel]
        for label_setter, letter in zip(axes_label_setters, ['X', 'Y', 'Z']):
            label_setter(f'{letter} Axis')

        for plane, color in zip(planes, colors):
            face = Poly3DCollection([plane])
            face.set_color(color)
            ax.add_collection3d(face)
        
        plt.show(block=False)

    def demo(self, pivot_strategy=None):
        ax = plt.gca()
        for sol in linear_program.LinearProgramSolver.solve_simplex_steps(self._linear_program, pivot_strategy=pivot_strategy):
            print(sol)
            if self._previous_point is not None:
                line_x, line_y, line_z = ([p1, p2] for (p1, p2) in zip(self._previous_point, sol.solution))
                ax.plot(line_x, line_y, line_z, color='black', linewidth=10)

            x, y, z = sol.solution
            ax.scatter(x, y, z, c='r', marker='o', s=64)
            plt.draw()
            self._previous_point = sol.solution
            input('Hit enter for next step')

        plt.show()
            

class KleeMintyPlotter(Simplex3DPlotter):
    @staticmethod
    def create():
        planes = np.array([
            # bottom x=0
            [(0, 0, 0), (0, 25, 0), (0, 25, 25), (0, 0, 125)],
            # y=0
            [(0, 0, 0), (5, 0, 0), (5, 0, 85), (0, 0, 125)],
            # z=0
            [(0, 0, 0), (5, 0, 0), (5, 5, 0), (0, 25, 0)],
            # top
            [(0, 25, 25), (5, 5, 65), (5, 0, 85), (0, 0, 125)],
            # front: x=5
            [(5, 0, 0), (5, 5, 0), (5, 5, 65), (5, 0, 85)],
            # side: x=5
            [(0, 25, 0), (0, 25, 25), (5, 5, 65), (5, 5, 0)],
        ])
        colors = np.array([
            'darkgrey',
            'skyblue',
            'deepskyblue',
            'cornflowerblue',
            'turquoise',
            'aqua'
        ])
        scale = np.array([[0, 5], [0, 25], [0, 125]])
        return Simplex3DPlotter(KleeMintyPlotter._generate_linear_program(), planes, colors, scale)

    @staticmethod
    def _generate_linear_program():
        objective_func = array([4, 2, 1])
        constraint_lhs = array([[1, 0, 0], [4, 1, 0], [8, 4, 1]])
        constraint_rhs = array([5, 25, 125])
        lp = linear_program.StandardLinearProgram(objective_func, constraint_lhs, constraint_rhs)
        return lp

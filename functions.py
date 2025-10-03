import logging
from datetime import datetime

import numpy as np
import structlog
import ufl
from dolfinx.fem import Function
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import MeshTags
from mpi4py import MPI
from ufl import TrialFunction, TestFunction
# from util.custom_handler import CustomHandler

from dolfinx.fem import (Expression, Function, FunctionSpace,
                         assemble_scalar, dirichletbc, form, locate_dofs_topological)


def save(u_func: Function, filename: str, mesh_tags: MeshTags = None, V: FunctionSpace = None, append: bool = False,
         time: float = None, close: bool = True, xdmf: XDMFFile = None):
    """
    Save an ufl function to a file by projection onto V
    :param xdmf:
    :param close:
    :param time:
    :param append:
    :param V:
    :param mesh_tags: optional: mesh tags to save to file
    :param u_func: The ufl function to save
    :param filename: The filename to store the function in
    :return: None
    """
    folder = "results/"
    if V is None:
        V = u_func.function_space
    mesh = V.mesh
    uh = Function(V)
    u = TrialFunction(V)
    v = TestFunction(V)
    A1 = ufl.inner(u, v) * ufl.dx
    L1 = ufl.inner(u_func, v) * ufl.dx
    problem = LinearProblem(A1, L1, u=uh, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    problem.solve()
    if xdmf is None:
        xdmf = XDMFFile(MPI.COMM_SELF, folder + filename, "w")
    if not append:
        xdmf.write_mesh(mesh)
        if mesh_tags:
            xdmf.write_meshtags(mesh_tags, mesh.geometry)
    if time is None:
        xdmf.write_function(uh)
    else:
        xdmf.write_function(uh, time)
    if close:
        xdmf.close()
    else:
        return xdmf


def function_norm(u_func: Function, norm: str = 'l2') -> float:
    if norm.lower() == "l2":
        error_form = form(ufl.inner(u_func, u_func) * ufl.dx)
    elif norm.lower() == "h1":
        error_form = form(ufl.inner(ufl.grad(u_func), ufl.grad(u_func)) * ufl.dx)
    else:
        raise NotImplementedError(f"Unknown norm {norm}")
    return np.sqrt(u_func.function_space.mesh.comm.allreduce(assemble_scalar(error_form)))


def petsc2array(matrix):
    """
    Extract data from a PETSc matrix as numpy array
    :param matrix:
    :return:
    """
    return matrix.getValues(range(0, matrix.getSize()[0]), range(0, matrix.getSize()[1]))

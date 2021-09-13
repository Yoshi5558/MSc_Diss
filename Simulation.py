from __future__ import print_function
from __future__ import absolute_import
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import numpy as nm
from numpy.linalg import norm
import pymesh
import sys
import os

from sfepy.postprocess.viewer import Viewer

sys.path.append('.')
from sfepy.base.base import IndexedStruct, Struct
from sfepy.discrete import (FieldVariable, Material, Integral,
                            Equation, Equations, Problem, Functions)
from sfepy.discrete.fem import Mesh, FEDomain, Field
from sfepy.terms import Term
from sfepy.discrete.conditions import Conditions, EssentialBC, InitialCondition
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton
from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
from sfepy.base.conf import transform_functions
from sfepy.solvers.ts_solvers import SimpleTimeSteppingSolver
from scipy.spatial import distance
import matplotlib.pyplot as plt

import six



helps = {
    'young': "the Young's modulus [default: %(default)s]",
    'poisson': "the Poisson's ratio [default: %(default)s]",
    'load': "the vertical load value (negative means compression)"
            " [default: %(default)s]",
    'show': 'show the results figure',
    'probe' : 'probe the results',
    'order' : 'displacement field approximation order [default: %(default)s]',
}

"!!Geometry Generation helper functions!!"
def generate_cube_mesh(dim_lim_min, dim_lim_max):
    #Define a cube mesh by a minimum and maximum corner coord
    cube = pymesh.generate_box_mesh(nm.array([dim_lim_min, dim_lim_min, dim_lim_min])
                                    , nm.array([dim_lim_max, dim_lim_max, dim_lim_max]), num_samples=32)
    return cube


def generate_sphere_mesh(radius, center_vector):
    #Define a sphere mesh of radius radius and centre center_vector
    sphere = pymesh.generate_icosphere(radius, nm.array(center_vector), refinement_order=3)
    return sphere


def generate_outgas_line(start_vector, end_vector):
    #Define a 3D rectangle as an outgas port between and minimum and maximum coord bound
    outgas_tube = pymesh.generate_box_mesh(nm.array(start_vector), nm.array(end_vector),num_samples=2)
    return outgas_tube


def custom_tetra(input_mesh, refinement):
    #Redefine an input triangular mesh input_mesh as tet elements with target edge length refinement
    tetra_out = pymesh.tetrahedralize(input_mesh, refinement, radius_edge_ratio=3.0, facet_distance=-0.5,
                                      feature_angle=30, engine='tetgen', with_timing=False)
    return tetra_out


def calculate_outgas(cavity_centre, outgas_width, direction='down'):
    #Calculate source vectors for the outgas port given higher characteristics
    if direction == 'down':
        return [-6, 0 - outgas_width / 2, 0 - outgas_width / 2], [cavity_centre[0], 0 + outgas_width / 2, 0 + outgas_width / 2]
       # return [-6, cavity_centre[1]-outgas_width/2, -outgas_width/2], [0, cavity_centre[1]+outgas_width/2, outgas_width/2]


def meshify_cavity(instructions, experiment_directory):
    """
    Function to generate the geometry for a given set of instructions
    """
    #Define geometry
    dim_lim = instructions[0][0] / 2
    cube = generate_cube_mesh(dim_lim, -dim_lim)
    sphere = generate_sphere_mesh(instructions[1][1], instructions[1][0])
    outgas_0, outgas_1 = calculate_outgas(instructions[1][0], 0.4)

    outgas_tube = generate_outgas_line(outgas_0, outgas_1)

    #Execute mesh-mesh operators
    cavity = pymesh.boolean(sphere, outgas_tube, operation="union")
    pymesh.save_mesh(experiment_directory + 'cavity.stl', cavity, ascii=True)
    output_mesh = pymesh.boolean(cube, cavity, operation="union", engine="cork")
    # Auto-refine mesh and convert to tetrahedral elements
    output_mesh = fix_mesh(output_mesh, 0.0144)
    output_mesh = custom_tetra(output_mesh, 0.4)
    #Save mesh as .msh file to experiment directory
    pymesh.save_mesh(experiment_directory + 'output.stl', output_mesh, ascii=True)
    pymesh.save_mesh(experiment_directory + 'output.msh', output_mesh, *output_mesh.get_attribute_names(), ascii=True)
    return experiment_directory + 'output.msh', instructions[2]


def define_experiment(box_dims, max_height, min_height, max_width, min_width, step, magnets):
    """
    Function to define entire search spaces through the use of bounds and steps.
    """
    #Define parameter spaces
    instructions = []
    y_space = nm.arange(-box_dims[0]/2, box_dims[0]/2, step)
    x_space = nm.arange(min_width, max_width, step)
    #Step through spaces and store experimental definition
    for y in y_space:
        for x in x_space:
            if not ((y > 0 and y + x > max_height) or (y < 0 and y - x < min_height)):
                for magnet in magnets:
                    entry = [[y, 0, 0], x]
                    instructions.append([box_dims, entry, magnet])
    return instructions

def generate_force_vector(target, num):
    """
    Simple helper function for the creation of standard force vectors
    """
    force_vector = []
    j = 0
    for i in range(0,num):
        if i < num:
            j += target / num
        force_vector.append(j)
    return force_vector


def experiment_1_controller(experiment_instructions, force_target, stepnum, exp_type):
    """
    Experiment 1 - Graphing the changes in Force / Displacement Relationship for LE or MR models
    :param experiment_instructions:
    :param force_vector:
    :return:
    """
    #Define experiment directory (pass if already defined)
    experiment_directory = os.getcwd() + '/user_meshes/'
    if not os.path.isdir(experiment_directory):
        os.mkdir(experiment_directory)
    score_arr, force_vector, results_data = [],generate_force_vector(force_target, stepnum),[]
    print("Calculated Experimental Force Vector:    ")
    print(force_vector)
    for experiment in experiment_instructions:
        print('Starting experiment: ' + str(experiment_instructions.index(experiment)) + '/' + str(len(experiment_instructions)))
        print('Job Instructions: ' + str(experiment))
        mesh_loc, magnet = meshify_cavity(experiment, experiment_directory)
        if exp_type == 'LE':
            u_vector = simulation_LE(mesh_loc, magnet, force_vector=force_vector)
        elif exp_type == 'MR':
            u_vector = simulation_MR(mesh_loc, magnet, force_vector=force_vector)
        local_arr = []
        for u in u_vector:
            local_arr.append([sum(x) / len(x) for x in zip(*u)][0])
        score_arr.append(local_arr)
    graph_ex_1(force_vector,score_arr, experiment_instructions)
    print(force_vector,score_arr, experiment_instructions)

def graph_ex_1(force_vector, data_vector, experiment_instructions):
    """
    Plots line graphs for simulation outputs, paired with experiment_1_controller
    """
    for data in data_vector:
        print(data)
        plt.plot(force_vector, data, label='Cavity Height: ' + str(experiment_instructions[data_vector.index(data)][1][0][1]) + 'Radius: ' + str(experiment_instructions[data_vector.index(data)][1][1]))
    plt.title('Force vs Magnet Displacement Experiment')
    plt.xlabel('Force Applied /N')
    plt.ylabel('Magnet Displacement / mm')
    plt.legend()
    plt.show()

def experiment_2_controller(experiment_instructions, force_target, stepnum, exp_type):
    """
    Experiment 2 - Scoring outcomes of EB creation
    :param experiment_instructions: Geometry Encoding Instructions
    :param force_target: The maximum load of the EB
    :param stepnum: The number of steps in the timestepping solver
    :param exp_type: The type of experiment 'LE' for linear elastic, 'MR' for Mooney-Rivlin
    :return:
    """
    #Define experiment directory (pass if already defined)
    experiment_directory = os.getcwd() + '/user_meshes/'
    if not os.path.isdir(experiment_directory):
        os.mkdir(experiment_directory)
    print("!!Single Shot Mode!! Experimental Force:    ")
    print(str(force_target) + 'N')
    for experiment in experiment_instructions:
        print('Starting experiment: ' + str(experiment_instructions.index(experiment)) + '/' + str(len(experiment_instructions)))
        print('Job Instructions: ' + str(experiment))
        mesh_loc, magnet = meshify_cavity(experiment, experiment_directory)
        if exp_type == 'LE':
            u_vector = simulation_LE(mesh_loc, magnet, force_vector=[force_target], visualiser=True)
        elif exp_type == 'MR':
            u_vector = simulation_MR(mesh_loc, magnet, force_vector=[force_target], visualiser=True)

def fix_mesh(mesh, tolerance):
    """
    Script to convert raw mesh outputs of boolean operations, made of 2_3 elements,
    to 3_4 elements whilst redefining the mesh to be equally divided.
    :param mesh: Mesh to fix
    :param tolerance: A float to control how fine the mesh division are (normally just set and forgotten)
    """
    bbox_min, bbox_max = mesh.bbox
    diag_len = norm(bbox_max - bbox_min)
    target_len = round(diag_len * tolerance, 5)
    print("Target resolution: {} mm".format(target_len))
    count = 0
    mesh, __ = pymesh.remove_degenerated_triangles(mesh, 100)
    mesh, __ = pymesh.split_long_edges(mesh, target_len)
    num_vertices = mesh.num_vertices
    while True:
        mesh, __ = pymesh.collapse_short_edges(mesh, 1e-6)
        mesh, __ = pymesh.collapse_short_edges(mesh, target_len,
                                               preserve_feature=True)
        mesh, __ = pymesh.remove_obtuse_triangles(mesh, 150.0, 100)
        if mesh.num_vertices == num_vertices:
            break

        num_vertices = mesh.num_vertices
        print("#v: {}".format(num_vertices))
        count += 1
        if count > 10:
            break

    mesh = pymesh.resolve_self_intersection(mesh)
    mesh, __ = pymesh.remove_duplicated_faces(mesh)
    mesh = pymesh.compute_outer_hull(mesh)
    mesh, __ = pymesh.remove_duplicated_faces(mesh)
    mesh, __ = pymesh.remove_obtuse_triangles(mesh, 179.0, 5)
    mesh, __ = pymesh.remove_isolated_vertices(mesh)

    return mesh


def inner_region_(coors, domain=None):
    """
    Magnet region definition (EB)
    """
    magnet = global_magnet
    x, y, z = coors[:, 0], coors[:, 1], coors[:, 2]
    mag_min_x, mag_max_x = (magnet[1][0] - (magnet[0][0])) / 2, (magnet[1][0] + (magnet[0][0])) / 2
    mag_min_y, mag_max_y = (magnet[1][1] - (magnet[0][1])) / 2, (magnet[1][1] + (magnet[0][1])) / 2
    mag_min_z, mag_max_z = (magnet[1][2] - (magnet[0][2])) / 2, (magnet[1][2] + (magnet[0][2])) / 2
    return nm.where((x > mag_min_x) & (x < mag_max_x) & (y > mag_min_y) & (y < mag_max_y) & (z > mag_min_z) & (z < mag_max_z))[0]


def outer_region_(coors, domain=None):
    """
    Non - magnet region definition (EB)
    """
    magnet = global_magnet
    x, y, z = coors[:, 0], coors[:, 1], coors[:, 2]
    mag_min_x, mag_max_x = (magnet[1][0] - (magnet[0][0] / 2)) / 2, (magnet[1][0] + (magnet[0][0] / 2)) / 2
    mag_min_y, mag_max_y = (magnet[1][1] - (magnet[0][1] / 2)) / 2, (magnet[1][1] + (magnet[0][1] / 2)) / 2
    mag_min_z, mag_max_z = (magnet[1][2] - (magnet[0][2] / 2)) / 2, (magnet[1][2] + (magnet[0][2] / 2)) / 2
    return nm.where((x < mag_min_x) | (x > mag_max_x) | (y < mag_min_y) | (y > mag_max_y)| (z < mag_min_z) | (z > mag_max_z))[0]


def store_u(displacements):
    """
    Post-Process Function for magnet region identification
    :param displacements: 
    """
    def _store(problem, ts, state):
        magnet_reg = problem.domain.regions['magnet0']
        u = problem.get_variables()['u'].get_state_in_region(magnet_reg)
        displacements.append(u)
    return _store


def solve_disp(problem):
    """
    Post-Process Function for magnet displacement calculation
    :param problem: Solved SFEpy Problem instance  
    """
    displacements = {}
    out = []
    problem.solve(save_results=False, step_hook=store_u(out))
    displacements['u'] = nm.array(out, dtype=nm.float64)
    return displacements

def simulation_LE(mesh_loc, magnet, force_vector, visualiser=False):
    """
        - Linear Elasticity FEA test - 
        :param mesh_loc: The location of a mesh, formatted as a .msh file 
        :param magnet: Location and dimensions of the magnet
        :param force_vector: A vector containing all the force increments you wish the timestepping solver to stop at
        :param visualiser: Boolean, toggles display mode
    """
    prev_force = 0
    first = True
    results = []
    for force in force_vector:
        parser = ArgumentParser(description=__doc__,
                                formatter_class=RawDescriptionHelpFormatter)
        parser.add_argument('--version', action='version', version='%(prog)s')

        parser.add_argument('--poisson', metavar='float', type=float, action='store', dest='poisson', default=0.45,
                            help=helps['poisson'])
        parser.add_argument('--young', metavar='float', type=float, action='store', dest='young', default=25e3,
                            help=helps['young'])
        parser.add_argument('-s', '--show', action="store_true", dest='show', default=False, help=helps['show'])
        options = parser.parse_args()

        global global_magnet
        global_magnet = magnet

        conf_functions = {
            'inner_region': (inner_region_,),
            'outer_region': (outer_region_,),
        }

        functions = Functions.from_conf(transform_functions(conf_functions))
        mesh = Mesh.from_file(mesh_loc)
        domain = FEDomain('domain', mesh)
        min_x, max_x = domain.get_mesh_bounding_box()[:, 0]
        eps = 0.01 * (max_x - min_x)

        omega = domain.create_region('Omega', 'all')
        gamma1 = domain.create_region('Gamma1', 'vertices in x < %.10f' % (min_x + eps), 'facet')
        gamma2 = domain.create_region('Gamma2', 'vertices in x > %.10f' % (max_x - eps), 'facet')
        magnet_region = domain.create_region('magnet0', 'vertices by inner_region', 'facet', functions=functions)
        soft_region = domain.create_region('elastomer0', 'vertices by outer_region', 'facet', functions=functions)

        m = Material('m', values={'D': {'elastomer0': stiffness_from_youngpoisson(3,options.young,options.poisson), 'magnet0': stiffness_from_youngpoisson(3, 200e3, 0.295)}})

        field = Field.from_args('fu', nm.float64, 'vector', omega, approx_order=1)

        u = FieldVariable('u', 'unknown', field)
        v = FieldVariable('v', 'test', field, primary_var_name='u')

        integral = Integral('i', order=3)

        load = Material('load', values={'val': force})

        t1 = Term.new('dw_lin_elastic(m.D, v, u)', integral, omega, m=m, v=v, u=u)
        t3 = Term.new('dw_surface_ltr( load.val, v )', integral, gamma2, load=load, v=v)
        eq = Equation('balance', t1 + t3)

        eqs = Equations([eq])

        fix = EssentialBC('fix', gamma1, {'u.all': 0.0})
        ic = InitialCondition('ic', gamma2, {'load.val': 0})


        ls = ScipyDirect({})
        nls_status = IndexedStruct()
        nls = Newton({}, lin_solver=ls, status=nls_status)

        pb = Problem('elasticity', equations=eqs)

        pb.set_bcs(ebcs=Conditions([fix]))
        pb.set_ics(Conditions([ic, ]))
        tss = SimpleTimeSteppingSolver({'t0': 0.0, 't1': 1.0, 'n_step': 2},
                                       context=pb,nls=nls)
        pb.set_solver(tss)
        u_u = solve_disp(pb)
        first = False
        prev_force = force
        results.append(nm.ndarray.tolist(u_u.get('u'))[1])
        if visualiser:
            view = Viewer('domain.1.vtk')
            view(vector_mode='warp_norm', rel_scaling=2,
                 is_scalar_bar=True, is_wireframe=True)
    print(results)
    return results

def simulation_MR(mesh_loc, magnet, force_vector, visualiser=False):
    """
    - Mooney-Rivlin FEA test - 
    :param mesh_loc: The location of a mesh, formatted as a .msh file 
    :param magnet: Location and dimensions of the magnet
    :param force_vector: A vector containing all the force increments you wish the timestepping solver to stop at
    :param visualiser: Boolean, toggles display mode
    """
    prev_force = 0
    first = True
    results = []
    for force in force_vector:
        parser = ArgumentParser(description=__doc__,
                                formatter_class=RawDescriptionHelpFormatter)
        parser.add_argument('-s', '--show', action="store_true", dest='show', default=False, help=helps['show'])
        options = parser.parse_args()
        global global_magnet
        global_magnet = magnet

        conf_functions = {
            'inner_region': (inner_region_,),
            'outer_region': (outer_region_,),
        }
        order = 3
        functions = Functions.from_conf(transform_functions(conf_functions))
        mesh = Mesh.from_file(mesh_loc)
        domain = FEDomain('domain', mesh)
        min_x, max_x = domain.get_mesh_bounding_box()[:, 0]
        eps = 0.01 * (max_x - min_x)

        omega = domain.create_region('Omega', 'all')
        gamma1 = domain.create_region('Gamma1', 'vertices in x < %.10f' % (min_x + eps), 'facet')
        gamma2 = domain.create_region('Gamma2', 'vertices in x > %.10f' % (max_x - eps), 'facet')
        magnet_region = domain.create_region('magnet0', 'vertices by inner_region', 'facet', functions=functions)
        soft_region = domain.create_region('elastomer0', 'vertices by outer_region', 'facet', functions=functions)

        c10, c01 = -0.98, 3.01
        m = Material(
            'm',k=1, mu=2 * c10, kappa=2 * c01,
        )

        vector_field = Field.from_args(
            'fu', nm.float64, 'vector', omega, approx_order=1)

        u = FieldVariable('u', 'unknown', vector_field, history=1)
        v = FieldVariable('v', 'test', vector_field, primary_var_name='u')

        integral0 = Integral('i', order=2)

        load = Material('load', values={'val': force})

        term_neohook = Term.new('dw_tl_he_neohook(m.mu, v, u)', integral0, omega, m=m, v=v, u=u)
        term_mooney = Term.new('dw_tl_he_mooney_rivlin(m.kappa, v, u)', integral0, omega, m=m, v=v, u=u)
        term_bulk = Term.new('dw_tl_bulk_penalty(m.k,v,u)', integral0, omega, m=m, v=v,u=u)
        term_volume = Term.new('dw_surface_ltr(load.val, v)', integral0, gamma2, load=load, v=v)

        eq_balance = Equation('balance', term_neohook + term_mooney + term_bulk- term_volume)
        eqs = Equations([eq_balance])

        fix = EssentialBC('fix', gamma1, {'u.all': 0.0})
        ic = InitialCondition('ic', gamma2, {'load.val': 0})


        ls = ScipyDirect({})
        nls_status = IndexedStruct()
        nls = Newton({}, lin_solver=ls, status=nls_status)

        pb = Problem('hyperelasticity', equations=eqs)

        pb.set_bcs(ebcs=Conditions([fix]))
        pb.set_ics(Conditions([ic, ]))
        tss = SimpleTimeSteppingSolver({'t0': 0.0, 't1': 1.0, 'n_step': 2},
                                       context=pb, nls=nls)
        pb.set_solver(tss)
        pb.solve(save_results=True)

        u_u = solve_disp(pb)
        print(u_u)
        prev_force = force
        results.append(nm.ndarray.tolist(u_u.get('u'))[0])
        if visualiser:
            view = Viewer('domain.1.vtk')
            view(vector_mode='warp_norm', rel_scaling=2,
                 is_scalar_bar=True, is_wireframe=True)
    print(results)
    return results


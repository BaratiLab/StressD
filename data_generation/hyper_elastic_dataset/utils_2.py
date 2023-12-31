import numpy as np
import matplotlib.pyplot as plt
from fenics import *
from dolfin import *
import pyvista as pv
import fenics as fe
import ufl
from utils import *
import meshio
import os

def in_polygon(x, y, center, radius, n, angle_offset=0):
    angle = np.arctan2(y-center[1], x-center[0]) + np.radians(angle_offset)
    return np.sqrt((x-center[0])**2 + (y-center[1])**2) <= radius * np.cos(np.pi/n) / np.cos(angle % (2*np.pi/n) - np.pi/n)


def generate_geometry(center, radius, n_sides, angle_offset=0,reso=32):
    x, y = np.meshgrid(np.linspace(0, 1, reso), np.linspace(0, 1, reso))
    mask = in_polygon(x, y, center, radius, n_sides, angle_offset)
    mask = mask.astype(int)

    inverted_mask = 1-mask

    return inverted_mask

def generate_mesh(img,name,plot=False):
    binary_array = img
    # Add a third dimension to the binary array
    binary_image = binary_array[np.newaxis, :, :]

    # Create an image data object

    grid = pv.wrap(binary_image.T)

    # Use threshold to create a mesh
    mesh = grid.threshold([0.5,1]).triangulate()

    num=len(mesh.split_bodies())

    # Visualize the mesh
    if plot:
        mesh.plot(show_edges=True)

    # num=len(mesh.split_bodies())

    if num>1:
        
        return False
    else:
        mesh.save(name+".vtk")
        mesh_io = meshio.read(name+".vtk")
        mesh_io.points=mesh_io.points[:,:2]
        meshio.write(name+".xdmf", mesh_io)
        
        # os.remove

        return True


def read_xdmf(name):
    mesh=Mesh()
    f=XDMFFile(name)
    f.read(mesh)
    return mesh


def bottom(x, on_boundary):
    return (on_boundary and fe.near(x[1], 0.0))


# Strain function
def epsilon(u):
    return 0.5*(fe.grad(u) + fe.grad(u).T)


# Stress function
def sigma(u):
    return lmbda*fe.div(u)*fe.Identity(2) + 2*mu*epsilon(u)


def simulate(mesh, plot_mesh=False,plot_displacement=False, plot_stress=False,E=20e5,mu=0.5,rho_0=200.0,g_int=1.0e6,b_int=-1000.0,name="test"):

    # Definition of Neumann condition domain
    boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)

    top = fe.AutoSubDomain(lambda x: fe.near(x[1], 63))

    top.mark(boundaries, 1)
    ds = fe.ds(subdomain_data=boundaries)

    # --------------------
    # Function spaces
    # --------------------
    V = fe.VectorFunctionSpace(mesh, "CG", 1)
    u_tr = fe.TrialFunction(V)
    u_test = fe.TestFunction(V)
    u = fe.Function(V)
    g = fe.Constant((0.0, g_int))
    b = fe.Constant((0.0, b_int))
    N = fe.Constant((0.0, 1.0))

    aa, bb, cc, dd, ee = 0.5*mu, 0.0, 0.0, mu, -1.5*mu

    # --------------------
    # Boundary conditions
    # --------------------
    bc = fe.DirichletBC(V, fe.Constant((0.0, 0.0)), bottom)

    # --------------------
    # Weak form
    # --------------------
    I = fe.Identity(2)
    F = I + fe.grad(u)  # Deformation gradient
    C = F.T*F  # Right Cauchy-Green tensor
    J = fe.det(F)  # Determinant of deformation fradient

    #psi = (aa*fe.tr(C) + bb*fe.tr(ufl.cofac(C)) + cc*J**2 - dd*fe.ln(J))*fe.dx - fe.dot(b, u)*fe.dx + fe.inner(f, u)*ds(1)
    n = fe.dot(ufl.cofac(F), N)
    surface_def = fe.sqrt(fe.inner(n, n))
    psi = (aa*fe.inner(F, F) + ee - dd*fe.ln(J))*fe.dx - rho_0*J*fe.dot(b, u)*fe.dx + surface_def*fe.inner(g, u)*ds(1)

    # --------------------
    # Solver
    # --------------------
    Form = fe.derivative(psi, u, u_test)
    Jac = fe.derivative(Form, u, u_tr)

    problem = fe.NonlinearVariationalProblem(Form, u, bc, Jac)
    solver = fe.NonlinearVariationalSolver(problem)
    prm = solver.parameters
    #prm["newton_solver"]["error_on_convergence"] = False
    #fe.solve(Form == 0, u, bc, J=Jac, solver_parameters={"error_on_convergence": False})
    solver.solve()

    print(np.amax(u.vector()[:]))

    # --------------------
    # Post-process
    # --------------------
    if plot_displacement:
        d=plot(u, mode="displacement")
        plt.colorbar(d)
        plt.show()


    # First Piola-Kirchhoff stress tensor
    P = mu*(F - fe.inv(F).T) + mu*fe.ln(J)*fe.inv(F).T

    # Compute Cauchy stress tensor
    sigma = 1.0/J*P*F.T

    # Project to function space
    sigma_proj = project(sigma, TensorFunctionSpace(mesh, 'P', 1))

    # To get the individual components of the stress tensor
    sigma_11, sigma_12, sigma_21, sigma_22 = sigma_proj.split(deepcopy=True)

    # Compute von Mises stress
    von_Mises = project(sqrt(sigma_11**2 - sigma_11*sigma_22 + sigma_22**2 + 3*sigma_12**2), FunctionSpace(mesh, 'P', 1))

    # # Plot von Mises stress
    # c=plot(von_Mises, title='Von Mises Stress')
    # # plt.show()

    if plot_stress:
        c=plot(von_Mises, title='Von Mises Stress')
        plt.colorbar(c)

        plt.show()

    # vtkfile_sigma = fe.File(name+'sigma.pvd')
    # vtkfile_sigma << sigma_proj

    # Save von Mises stress to a VTK file
    vtkfile_von_Mises = fe.File(name+'von_Mises.pvd')
    vtkfile_von_Mises << von_Mises

    vtk_displacement = fe.File(name+'displacement.pvd')
    vtk_displacement << u


def get_2d_res(name,reso,min_max=False):
    res_v = pv.read(name+"von_Mises.pvd")

    x=np.linspace(0,reso-1,reso)
    y=np.linspace(0,reso-1,reso)

    x,y=np.meshgrid(x,y)
    z = np.zeros_like(x)

    grid = pv.StructuredGrid(y, x, z)

    grid_with_val = grid.interpolate(res_v["Block-00"])
    vv=grid_with_val[res_v["Block-00"].array_names[0]].reshape((reso, reso))

    vv=np.array(vv)

    if min_max:
        print("Min: ",vv.min(),"Max: ",vv.max())

    return vv


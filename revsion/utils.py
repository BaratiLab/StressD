import numpy as np
import pyvista as pv
import meshio
from fenics import *
from dolfin import *
import fenics as fe
import ufl
import matplotlib.pyplot as plt

def make_mesh(center, radius, normal, n_sides):

    plane = pv.Plane(center=(0.5,0.5, 0.0),i_resolution=64, j_resolution=64,i_size=1.0, j_size=1.0)
    plane= plane

    polyG= pv.Polygon(center=(0.5,0.5, -0.5), radius=radius, n_sides=n_sides,normal=normal)
    polyG=polyG.extrude(normal,capping=True)

    # polyG = polyG.subdivide(2,"loop")


    mesh= plane.clip_surface(polyG, invert = True)
    mesh=mesh.triangulate()

    return mesh,polyG


def save_xdmf(mesh):
    # Convert PyVista mesh to meshio format
    faces = mesh.faces.reshape((-1, 4))[:, 1:]
    meshio_mesh = meshio.Mesh(mesh.points, {"triangle": faces})

    # Retain only the first two coordinates
    meshio_mesh.points = meshio_mesh.points[:,:2]
    
    # Write to .xdmf file directly
    meshio.write("m_t.xdmf", meshio_mesh)



def bottom(x, on_boundary):
    return (on_boundary and fe.near(x[1], 0.0))


# Strain function
def epsilon(u):
    return 0.5*(fe.grad(u) + fe.grad(u).T)


# Stress function
def sigma(u):
    return lmbda*fe.div(u)*fe.Identity(2) + 2*mu*epsilon(u)


def simulate(mesh, plot_mesh=False,plot_displacement=False, plot_stress=False,E=20e5,mu=0.5,rho_0=200.0,g_int=1.0e6,b_int=-1000.0):

    # f = XDMFFile('m_t.xdmf')
    # mesh=Mesh()
    # f.read(mesh)
    # if plot_mesh:
    #     plot(mesh)
    #     plt.show()

    # E = 20.0e5
    # mu = 0.5*E
    # rho_0 = 200.0

    # # Load
    # g_int = 1.0e6
    # b_int = -1000
    # .0

    # Definition of Neumann condition domain
    boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)

    top = fe.AutoSubDomain(lambda x: fe.near(x[1], 1.0))

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

    vtkfile_sigma = fe.File('sigma.pvd')
    vtkfile_sigma << sigma_proj

    # Save von Mises stress to a VTK file
    vtkfile_von_Mises = fe.File('von_Mises.pvd')
    vtkfile_von_Mises << von_Mises

        
        


    

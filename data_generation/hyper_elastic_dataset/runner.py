import numpy as np
import matplotlib.pyplot as plt
from fenics import *
from dolfin import *
import pyvista as pv
import fenics as fe
import ufl
from utils_2 import *
from itertools import product
from tqdm import tqdm
from multiprocessing import Pool


c_x = np.linspace(0.2, 0.5, 5)
c_y = np.linspace(0.2, 0.5, 5)


radius_l = np.linspace(0.1, 0.5, 10)

angle_l = np.linspace(0, 180, 5)
print(angle_l)
B= [0]

G=np.linspace(100,400,10)
print(G)


n_list = [3,4,5,6,7,8]


combinations = list(product(c_x, c_y, radius_l, angle_l, n_list,B,G))
print("Total combinations: ", len(combinations))

cnt_slv=[]
done = []

def runner(combi):
    
    img=generate_geometry(center=[combi[0], combi[1]], radius=combi[2], n_sides=combi[4], angle_offset=combi[3],reso=64)
    name = "_".join([str(ele).replace('.', 'p') if isinstance(ele, float) else str(ele) for ele in combi]).replace("p", ".")
    c=generate_mesh(img,name="./meshes/"+name,plot=False) 

    if c:  
        try:
            mesh_l = read_xdmf("./meshes/"+name+".xdmf")
            simulate(mesh_l,plot_displacement=False,plot_stress=False,plot_mesh=False,E=69e9,mu=50e5,rho_0=200.0,g_int=combi[6],b_int=combi[5],name="../r_data/fe_results_new_8/"+name)

            v=get_2d_res("../r_data/fe_results_new_8/"+name,reso=64)

            lc1 = np.zeros_like(v)
            lc2 = img

            #make top as g_int
            lc1[0,:]=combi[6]
            lc2 =lc2*combi[5]

            r= np.stack([img,lc1,lc2,v],axis=0)
            # print(r.shape)

            np.save("./results_8/"+name+".npy",r)

            os.remove("./meshes/"+name+".vtk")
            os.remove("./meshes/"+name+".xdmf")
            os.remove("./meshes/"+name+".h5")
            done.append(combi)
            np.save("done_new_8.npy",done)
        
            return r

        except:
            cnt_slv.append(combi)
            #save as txt file
            with open("cant_solve_new"+".txt", "w") as f:
                for s in cnt_slv:
                    f.write(str(s) +"\n")


###### Single thread ######
# for i in tqdm(range(len(combinations))):
#     r=runner(combinations[i])
    
#     break

###### Multi thread ######
if __name__ == '__main__':
    with Pool(16) as p:
        results = list(tqdm(p.imap(runner, combinations), total=len(combinations)))
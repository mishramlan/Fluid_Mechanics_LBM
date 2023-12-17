import numpy as np
from dataclasses import dataclass
from mpi4py import MPI
import matplotlib.pyplot as plt
# import ipyparallel as ipp
import psutil
import time

# only vars
velocity_set = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],
                         [0,0,1,0,-1,1,1,-1,-1]]).T
cores = psutil.cpu_count(logical= False)

# pack stuff so its together
@dataclass
class boundariesApplied:
    # left right top bottom
    apply_left : bool = False
    apply_right: bool = False
    apply_top:bool = False
    apply_bottom: bool = False

@dataclass
class cellNeighbors:
    left: int = -1
    right: int = -1
    top: int = -1
    bottom: int = -1

@dataclass
class mpiPackageStructure:
    # apply for boundaries
    boundaries_info: boundariesApplied = (False,False,False,False)
    neighbors: cellNeighbors = (-1, -1, -1, -1)
    # sizes and position in the whole grid
    Nx: int = 0
    Ny: int = 0
    pos_x : int = 0
    pos_y : int = 0
    # for MPI stuff
    rank : int = 0
    size : int = 0
    # overall
    relaxation : int = 0
    base_f: int = 0
    steps : int = 0
    wall_vel : int = 0

# set functions for the mpi Package Structure
def set_boundary_info(pox,poy,max_x,max_y):
    info = boundariesApplied(False,False,False,False)
    ##
    if pox == 0:
        info.apply_left = True
    if poy == 0:
        info.apply_bottom = True
    if pox == max_x:
        info.apply_right = True
    if poy == max_y:
        info.apply_top = True
    ##
    # print(info)
    return info

def get_postions_out_of_rank_size_quadratic(rank,size):
    ##
    # assume to be quadratic
    edge_lenght = int(np.sqrt(size))
    ###
    pox = rank % edge_lenght
    poy = rank//edge_lenght
    ###
    return pox,poy


def fill_mpi_struct_fields(rank,size,max_x,max_y,base_f,relaxation,steps,wall_vel):
    #
    info = mpiPackageStructure()
    info.rank = rank
    info.size = size
    info.pos_x,info.pos_y = get_postions_out_of_rank_size_quadratic(rank,size)
    info.boundaries_info = set_boundary_info(info.pos_x,info.pos_y,max_x-1,max_y-1) # i should know my own code lol
    info.Nx = base_f//(max_x) + 2
    info.Ny = base_f //(max_y) + 2
    info.neighbors = determin_neighbors(rank,size)
    #
    info.relaxation = relaxation
    info.base_f = base_f
    info.steps = steps
    info.wall_vel = wall_vel
    return info

def determin_neighbors(rank,size):
    # determin edge lenght
    edge_lenght = int(np.sqrt(size))
    ###
    neighbor = cellNeighbors()
    neighbor.top = rank + edge_lenght
    neighbor.bottom = rank - edge_lenght
    neighbor.right = rank + 1
    neighbor.left = rank-1
    ###
    return neighbor

# main methods
def stream(f):
    for i in range(1,9):
        f[i] = np.roll(f[i],velocity_set[i], axis = (0,1))

def equilibrium(rho_eq, ux_eq, uy_eq): #equilibrium approximation
    cu5 = ux_eq + uy_eq #velocity for channel 5
    cu6 = -ux_eq + uy_eq #velocity for channel 6
    cu7 = -ux_eq - uy_eq #velocity for channel 7
    cu8 = ux_eq - uy_eq #velocity for channel 8
    uu_eq = ux_eq**2 + uy_eq**2 #sum of the x and y squared velocities
    #returns feq as an array using the discretized equilibrium function
    weights = np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36]) # weights for each channel
    return np.array([weights[0]*rho_eq*(1 - 3/2*uu_eq), 
                     weights[1]*rho_eq*(1 + 3*ux_eq + 9/2*ux_eq**2 - 3/2*uu_eq),
                     weights[2]*rho_eq*(1 + 3*uy_eq + 9/2*uy_eq**2 - 3/2*uu_eq),
                    weights[3]*rho_eq*(1 - 3*ux_eq + 9/2*ux_eq**2 - 3/2*uu_eq),
                     weights[4]*rho_eq*(1 - 3*uy_eq + 9/2*uy_eq**2 - 3/2*uu_eq),
                     weights[5]*rho_eq*(1 + 3*cu5 + 9/2*cu5**2 - 3/2*uu_eq),
                     weights[6]*rho_eq*(1 + 3*cu6 + 9/2*cu6**2 - 3/2*uu_eq),
                     weights[7]*rho_eq*(1 + 3*cu7 + 9/2*cu7**2 - 3/2*uu_eq),
                     weights[8]*rho_eq*(1 + 3*cu8 + 9/2*cu8**2 - 3/2*uu_eq)])


def collision(f,rho,ux,uy,relaxation):
    f -= relaxation * (f - equilibrium(rho, ux, uy))


def caluculate_rho_ux_uy(f):
    rho = np.sum(f, axis=0)  # sums over each one individually
    ux = ((f[1] + f[5] + f[8]) - (f[3] + f[6] + f[7])) / rho
    uy = ((f[2] + f[5] + f[6]) - (f[4] + f[7] + f[8])) / rho
    return rho,ux,uy




def bounce_back_choosen(f,wall_vel,info):
    # modification for the bounce back for the bigger fs
    # Right + Left
    if info.boundaries_info.apply_right:
        # right so x = -1
        f[1, 1, :] = f[3, 0, :]
        f[5, 1, :] = f[7, 0, :]
        f[8, 1, :] = f[6, 0, :]
    if info.boundaries_info.apply_left:
        # left so x = 0
        f[3, -2, :] = f[1, -1, :]
        f[6, -2, :] = f[8, -1, :]
        f[7, -2, :] = f[5, -1, :]

    # Bottom + Top
    if info.boundaries_info.apply_bottom:
        # for bottom y = 0
        max_Ny = f.shape[2]-1
        f[2, :, 1] = f[4, :, 0]
        f[5, :, 1] = f[7, :, 0]
        f[6, :, 1] = f[8, :, 0]
    if info.boundaries_info.apply_top:
        # for top y = -1
        f[4, :, max_Ny-1] = f[2, :, max_Ny]
        f[7, :, max_Ny-1] = f[5, :, max_Ny] - 1 / 6 * wall_vel
        f[8, :, max_Ny-1] = f[6, :, max_Ny] + 1 / 6 * wall_vel


def comunicate(f,info,comm):
    # if they are false we have to comunicate otherwise will have to do the boundary stuff
    # Right + Left
    if not info.boundaries_info.apply_right:
        recvbuf = f[:, -1, :].copy()
        comm.Sendrecv(f[:,-2, :].copy(), info.neighbors.right, recvbuf=recvbuf, sendtag = 11, recvtag = 12)
        f[:, -1, :] = recvbuf
    if not info.boundaries_info.apply_left:
        recvbuf = f[:, 0, :].copy()
        comm.Sendrecv(f[:, 1, :].copy(), info.neighbors.left, recvbuf=recvbuf, sendtag = 12, recvtag = 11)
        f[:, 0, :] = recvbuf

    # Bottom + Top
    if not info.boundaries_info.apply_bottom:
        recvbuf = f[:,: ,0 ].copy()
        comm.Sendrecv(f[:, :,1 ].copy(), info.neighbors.bottom, recvbuf=recvbuf, sendtag = 99, recvtag = 98)
        f[:, :, 0] = recvbuf
    if not info.boundaries_info.apply_top:
        recvbuf = f[:, :, -1].copy()
        comm.Sendrecv(f[:, :, -2].copy(), info.neighbors.top, recvbuf=recvbuf, sendtag = 98, recvtag = 99)
        f[:, :, -1] = recvbuf


def collapse_data(process_info,f,comm):
    full_f = np.zeros(2)
    # process 0 gets the data and does the visualization
    if process_info.rank == 0:
        full_f = np.ones((9, process_info.base_f, process_info.base_f))
        original_x = process_info.Nx-2 # ie the base size of the f on that the
        original_y = process_info.Ny-2 # calculation ran
        # write the own stuff into it first
        full_f[:,0:original_x,0:original_y] = f[:,1:-1,1:-1]
        temp = np.zeros((9,original_x,original_y))
        for i in range(1,process_info.size):
            comm.Recv(temp,source = i,tag = i)
            x,y = get_postions_out_of_rank_size_quadratic(i,process_info.size)
            # determin start end endpoints to copy to in the f
            copy_start_x = 0 + original_x*x
            copy_end_x = original_x + original_x*x
            copy_start_y = 0 + original_y*y
            copy_end_y = original_y + original_y*y
            # copy
            full_f[:,copy_start_x:copy_end_x,copy_start_y:copy_end_y] = temp
    # all the others send to p0
    else:
        comm.Send(f[:,1:-1,1:-1].copy(),dest=0, tag = process_info.rank)

    return full_f

def sliding_lid_mpi(process_info,comm):
    #create f based on process Info
    rho = np.ones((process_info.Nx, process_info.Ny))
    ux = np.zeros((process_info.Nx, process_info.Ny))
    uy = np.zeros((process_info.Nx, process_info.Ny))
    f = equilibrium(rho, ux, uy)
    # loop
    for i in range(process_info.steps):
        stream(f)
        bounce_back_choosen(f, process_info.wall_vel, process_info)
        rho, ux, uy = caluculate_rho_ux_uy(f)
        collision(f, rho, ux, uy, process_info.relaxation)
        comunicate(f,process_info,comm)

    # get full f + plot
    full_f = collapse_data(process_info, f, comm)
    plotter(full_f,process_info)

def plotter(full_f,process_info):
    #plot
    if process_info.rank == 0:
        print("Making Image")
        # recalculate ux and uy
        idk,full_ux,full_uy = caluculate_rho_ux_uy(full_f)
        # acutal plot

        x = np.arange(0, process_info.base_f)
        y = np.arange(0, process_info.base_f)
        X, Y = np.meshgrid(x, y)
        speed = np.sqrt(full_ux.T ** 2 + full_uy.T ** 2)
        # plot
        plt.streamplot(X, Y, full_ux.T, full_uy.T, color=speed, cmap=plt.cm.jet)
        ax = plt.gca()
        ax.set_xlim([0, process_info.base_f + 1])
        ax.set_ylim([0, process_info.base_f + 1])
        plt.title("Sliding Lid")
        plt.xlabel("x-Position")
        plt.ylabel("y-Position")
        fig = plt.colorbar()
        fig.set_label("Velocity u(x,y,t)", rotation=270, labelpad=15)
        #savestring = "slidingLidmpi"+str(process_info.size)+".png"
        #plt.savefig(savestring)
        plt.show()

    if process_info.rank == 0:
        savestring = "slidingLidmpi"+str(process_info.size)+".txt"
        f = open(savestring,"w")
        totaltime = time.time() - startime
        f.write(str(totaltime))
        f.close()



def call():
    # vars
    steps = 100
    re = 1000
    base_lenght = 300
    wall_vel = 0.1
    relaxation = (2 * re) / (6 * base_lenght * wall_vel + re)
    # calls
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank_in_one_direction = int(np.sqrt(size)) # for an MPI thingi with 9 processes -> 3x3 field
    if rank_in_one_direction*rank_in_one_direction != size:
        return RuntimeError
    process_info = fill_mpi_struct_fields(comm.Get_rank(),size,
                                          rank_in_one_direction,rank_in_one_direction,base_lenght,
                                          relaxation,steps,wall_vel)
    # print(process_info)
    print(process_info.rank,process_info.size)
    sliding_lid_mpi(process_info,comm)


# call()
#comm = MPI.COMM_WORLD
#process_info = fill_mpi_struct_fields(0,4,2,2,300,0,0,0)
#fg = collapse_data(process_info,np.ones((9,152,152)),comm)
#print(fg.shape)
# g = np.zeros((9,27,27))
# k = g[:,1:-1,1:-1]
# print(k.shape)
startime = time.time()
call()
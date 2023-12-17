import numpy as np
import matplotlib.pyplot as plt

''' Setup '''


omega = 0.5
Nx = 100
Ny = 50
c = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],
                         [0,0,1,0,-1,1,1,-1,-1]]).T
rho_null = 1
diff = 0.001
shear_viscosity = (1/omega-0.5)/3


def stream(f):
    for i in range(1,9):
        f[i] = np.roll(f[i],c[i], axis = (0,1))



#def equilibrium(rho,ux,uy):
    #uxy = 3 * (ux + uy)
    #uu =  3 * (ux * ux + uy * uy)
    #ux_6 = 6*ux
    #uy_6 = 6*uy
    #uxx_9 = 9 * ux*ux
    #uyy_9 = 9 * uy*uy
    #uxy_9 = 9 * ux*uy
    #return np.array([(2 * rho / 9) * (2 - uu),
                     #(rho / 18) * (2 + ux_6 + uxx_9 - uu),
                     #(rho / 18) * (2 + uy_6 + uyy_9 - uu),
                     #(rho / 18) * (2 - ux_6 + uxx_9 - uu),
                     #(rho / 18) * (2 - uy_6 + uyy_9 - uu),
                     #(rho / 36) * (1 + uxy + uxy_9 + uu),
                     #(rho / 36) * (1 - uxy - uxy_9 + uu),
                     #(rho / 36) * (1 - uxy + uxy_9 + uu),
                     #(rho / 36) * (1 + uxy - uxy_9 +uu)])


def equilibrium_array(rho_eq, ux_eq, uy_eq): #equilibrium approximation
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



def calculate_real_values(f):
    rho = np.sum(f, axis=0)
    ux = ((f[1] + f[5] + f[8]) - (f[3] + f[6] + f[7])) / rho
    uy = ((f[2] + f[5] + f[6]) - (f[4] + f[7] + f[8])) / rho
    return rho, ux, uy

def collision(f,rho,ux,uy):
    # calculate equilibrium + apply collision
    f -= omega * (f-equilibrium_array(rho, ux, uy))



def bounce_back(f,wall_vel):
    # baunce back without any velocity gain
    # TODO rho Wall missing
    max_Ny = f.shape[2]-1  # y
    # for bottom y = 0
    f[2, :, 1] = f[4, :, 0]
    f[5, :, 1] = f[7, :, 0]
    f[6, :, 1] = f[8, :, 0]
    #f[4, :, 0] = 0
    #f[7, :, 0] = 0
    #f[8, :, 0] = 0
    # for top y = max_Ny
    f[4, :, max_Ny-1] = f[2, :, max_Ny]
    f[7, :, max_Ny-1] = f[5, :, max_Ny] - 1 / 6 * wall_vel
    f[8, :, max_Ny-1] = f[6, :, max_Ny] + 1 / 6 * wall_vel
    #f[2, :, max_Ny] = 0
    #f[5, :, max_Ny] = 0
    #f[6, :, max_Ny] = 0

def periodic_boundary_with_pressure_variations(f,rho_in,rho_out):
    # get all the values
    max_Nx = f.shape[1]-1
    rho, ux, uy = calculate_real_values(f)
    equilibrium = equilibrium_array(rho, ux, uy)
    ##########
    equilibrium_in = equilibrium_array(rho_in, ux[max_Nx-1,:], uy[max_Nx-1, :])
    # inlet 1,5,8
    f[:, 0, :] = equilibrium_in + (f[:, max_Nx-1, :] - equilibrium[:, max_Nx-1, :])

    # outlet 3,6,7
    equilibrium_out = equilibrium_array(rho_out, ux[1, :], uy[1, :])
    # check for correct sizes
    f[:, max_Nx, :] = equilibrium_out + (f[:, 1, :] - equilibrium[:, 1, :])


''' body '''

def poiseuille_flow():
    # main code
    print("Poiseuille Flow")
    uw = 0.000
    steps = 4000 # crashes 4533
    rho_in = rho_null+diff
    rho_out = rho_null-diff
    # initialize
    rho = np.ones((Nx+2, Ny + 2))
    ux = np.zeros((Nx+2, Ny + 2))
    uy = np.zeros((Nx+2, Ny + 2))
    f = equilibrium_array(rho, ux, uy)

    # loop
    for i in range(steps):
        periodic_boundary_with_pressure_variations(f,rho_in,rho_out)
        stream(f)
        bounce_back(f, uw)
        rho, ux, uy = calculate_real_values(f)
        collision(f, rho, ux, uy)

    # visualize
    delta = 2.0 * diff /Nx / shear_viscosity / 2.
    y = np.linspace(0, Ny, Ny+1) + 0.5
    u_analytical = delta * y * (Ny - y) / 3.
    plt.plot(u_analytical[:-1], label='Analytical')
    # plt.plot(u_analytical, label='analytical')
    #number_of_cuts_in_x = 4
    #for i in range(1,number_of_cuts_in_x):
    #point = int(Nx/number_of_cuts_in_x)
    plt.plot(ux[int(Nx-1), 1:-1], label = "Calculated @ x=100" )
    plt.plot(ux[int(Nx/2), 1:-1], label = "Calculated @ x=50" )
    plt.plot(ux[int(1), 1:-1], label = "Calculated @ x=1" )
    plt.plot(ux[int(0), 1:-1], label = "Calculated @ x=0" )
    print(len(ux[int(25), 1:-1]))
    plt.legend()
    plt.xlabel('Position in cross section')
    plt.ylabel('Velocity [m/s]')
    plt.title('Pouisuelle flow')
    plt.show()

def poiseuille_flow_fancy():
    # main code
    print("Poiseuille Flow fancy")
    runs = 10
    uw = 0.000
    steps = 4000  # crashes 4533
    rho_in = rho_null + diff
    rho_out = rho_null - diff
    # initialize
    rho = np.ones((Nx + 2, Ny + 2))
    ux = np.zeros((Nx + 2, Ny + 2))
    uy = np.zeros((Nx + 2, Ny + 2))
    f = equilibrium_array(rho, ux, uy)

    # plot related stuff
    x = np.arange(0, Nx)
    y = np.arange(0, Ny)
    X, Y = np.meshgrid(x, y)

    # loop
    for k in range(runs):
        uw += 0.001
        uw = round(uw,3)
        for i in range(steps):
            periodic_boundary_with_pressure_variations(f, rho_in, rho_out)
            stream(f)
            bounce_back(f, uw)
            rho, ux, uy = calculate_real_values(f)
            collision(f, rho, ux, uy)
            point = int(Nx/2)
            #
        plt.plot(ux[point, 1:-1], label="uw = {}".format(uw))
    ## end plot
    plt.legend()
    plt.xlabel('Position in cross section')
    plt.ylabel('Velocity [m/s]')
    plt.title('Pouisuelle flow with diffrent u-Walls')
    savestring = "PouisuelleFlowFancy.png"
    plt.savefig(savestring)
    plt.show()

poiseuille_flow()
#poiseuille_flow_fancy()
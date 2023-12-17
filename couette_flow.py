import numpy as np
import matplotlib.pyplot as plt

''' Setup '''


omega = 1.7
Nx = 50
Ny = 50
c = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],
                         [0,0,1,0,-1,1,1,-1,-1]]).T


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



def calculate_real_values(f):
    rho = np.sum(f, axis=0)
    ux = ((f[1] + f[5] + f[8]) - (f[3] + f[6] + f[7])) / rho
    uy = ((f[2] + f[5] + f[6]) - (f[4] + f[7] + f[8])) / rho
    return rho, ux, uy

def collision(f,rho,ux,uy):
    # calculate equilibrium + apply collision
    f -= omega * (f-equilibrium(rho, ux, uy))



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

vari =[]

''' body '''
def couette_flow():
    # main code
    print("couette Flow")
    steps = 1000
    plot_every = 500
    wall_vel = 0.01

    # initialize
    rho = np.ones((Nx,Ny))
    ux = np.zeros((Nx, Ny))
    uy = np.zeros((Nx,Ny))
    f = equilibrium(rho,ux,uy)

    # loop
    for i in range(steps):
        rho, ux, uy = calculate_real_values(f)
        collision(f,rho,ux,uy)
        stream(f)
        bounce_back(f,wall_vel)
        if (i % plot_every) == 0:
            plot_every = plot_every+1000
            #vari.append(np.average(rho[-2,:]))
            plt.plot(ux[Nx-1, 1:-1], label = "Step {}".format(i))

    # visualize
    x = np.arange(0,48)
    y = wall_vel*1/48*x
    
    plt.plot(y, label ="Analytical")
    plt.legend()
    plt.xlabel('Position in cross section')
    plt.ylabel('Velocity [m/s]')
    plt.title('Couette flow')
    plt.show()

   

couette_flow()


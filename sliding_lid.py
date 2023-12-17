import numpy as np
import matplotlib.pyplot as plt

# initial variables and sizess
re = 1000
base_lenght = 300
steps = 50000
wall_vel = 0.1
Nx = base_lenght
Ny = base_lenght
relaxation = (2*re)/(6*base_lenght*wall_vel+re)
c = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],
                         [0,0,1,0,-1,1,1,-1,-1]]).T
# main methods
def stream(f):
    for i in range(1,9):
        f[i] = np.roll(f[i],c[i], axis = (0,1))


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


def collision(f,rho,ux,uy):
    f -= relaxation * (f - equilibrium(rho, ux, uy))


def caluculate_rho_ux_uy(f):
    rho = np.sum(f, axis=0)  # sums over each one individually
    ux = ((f[1] + f[5] + f[8]) - (f[3] + f[6] + f[7])) / rho
    uy = ((f[2] + f[5] + f[6]) - (f[4] + f[7] + f[8])) / rho
    return rho,ux,uy


def bounce_back(f,uw):

    #### Left + Right
    # right so x = 0
    #max_Nx = f.shape[1]-1
    f[1, 1, :] = f[3, 0, :]
    f[5, 1, :] = f[7, 0, :]
    f[8, 1, :] = f[6, 0, :]
    # left so x = -1
    f[3, -2, :] = f[1, -1, :]
    f[6, -2, :] = f[8, -1, :]
    f[7, -2, :] = f[5, -1, :]

    #### TOP + Bottom
    # TODO rho_wall
    # for bottom y = 0
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

# body
def sliding_lid():
    print("Sliding Lid")

    # initizlize the gird
    rho = np.ones((Nx+2,Ny+2))
    ux = np.zeros((Nx+2,Ny+2))
    uy = np.zeros((Nx+2,Ny+2))
    f = equilibrium(rho,ux,uy)

    # loop
    for i in range(steps):
        stream(f)
        bounce_back(f,wall_vel)
        rho, ux, uy = caluculate_rho_ux_uy(f)
        collision(f,rho,ux,uy)

    # print(f[2,0,:])
    # visualize
    # values
    x = np.arange(0, Nx)
    y = np.arange(0, Ny)
    X, Y = np.meshgrid(x, y)
    speed = np.sqrt(ux[1:-1,1:-1].T ** 2 + uy[1:-1,1:-1].T ** 2)
    # plot
    plt.streamplot(X,Y,ux[1:-1,1:-1].T,uy[1:-1,1:-1].T, color = speed, cmap= plt.cm.jet)
    ax = plt.gca()
    ax.set_xlim([0, 301])
    ax.set_ylim([0, 301])
    plt.title("Sliding Lid")
    plt.xlabel("x-Position")
    plt.ylabel("y-Position")
    fig = plt.colorbar()
    fig.set_label("Velocity u(x,y,t)", rotation=270,labelpad = 15)
    plt.show()
    '''
    plt.plot(ux[int(1 + Nx / 2), 1:-1], color="green")
    plt.xlabel('Position in cross section')
    plt.ylabel('velocity')
    plt.title('Constant velocity')
    plt.show()
    '''



# call
sliding_lid()
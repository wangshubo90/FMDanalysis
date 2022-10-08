import numpy as np
from scipy import special
from scipy.optimize import least_squares
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import cv2

def sigmoid(x):
    
    return 1/(1+np.exp(-2*x))

def calc_edge_fourier(mesh_x, a, N=5):
    '''
        f = a_0 + \sum_{n=1}^{N} a_n * cos(w * n * x) + b_n * sin(w * n * x),
        where
        a_0 = a[0]
        w = a[1]
        a_n = a[n+1]
        b_n = a[n+2]
        
        a.shape = 2N+2
    '''
    
    edge1 = a[0]
    w1 = a[1]
    edge2 = a[2*N+2]
    w2 = a[2*N+3]
    
    for i in range(1, N+1):
        edge1 += a[i*2] * np.cos(w1 * i * mesh_x) + a[i*2+1] * np.sin(w1 * i *mesh_x)
        edge2 += a[i*2+2*N+2] * np.cos(w2 * i * mesh_x) + a[i*2+3+2*N] * np.sin(w2 * i *mesh_x)
    
    return edge1, edge2

def fourier_init_condition(H, N=5):
    guess = np.array([H*0.25, 0.3, *(0.1,)*2*N, H*0.75, 0.3, *(0.1,)*2*N])
    bounds = (np.array([0, 0.1,*(-np.inf,)*2*N, H//2, 0.2, *(-np.inf,)*2*N]), np.array([H//2, 3, *(np.inf,)*2*N, H, 0.7, *(np.inf,)*2*N]))
    return guess, bounds

def calc_edge_polynomial(mesh_x, a, order=2):
    
    edge1 = a[0]
    edge2 = a[order+1]
    
    for i in range(1, order+1):
        edge1 += a[i]*mesh_x**i
        edge2 += a[i+order+1]*mesh_x**i
    
    return edge1, edge2

def calc_map(a, mesh_x, mesh_y, edge_func=calc_edge_polynomial, **kargs):
    edge1, edge2 = edge_func(mesh_x, a, **kargs)
    
    # z1 = -special.erf((mesh_y - edge1) * 1)
    # z2 = special.erf((mesh_y - edge2) * 1)
    
    z1 = -sigmoid((mesh_y - edge1)*0.5)
    z2 =  sigmoid((mesh_y - edge2)*0.5)
    
    z = z1 + z2
    
    return z

def calc_thickness(a, mesh_x, edge_func=calc_edge_fourier):
    
    edge1, edge2 = edge_func(mesh_x, a)
    thickness = np.mean(np.abs(edge1 - edge2))
    
    return thickness

def masking(image, a, edge_func=calc_edge_polynomial, **kargs):
    h, w = image.shape
    x_grid = np.linspace(0, w-1, w)
    y_grid = np.linspace(0, h-1, h)
    mesh_x, mesh_y = np.meshgrid(x_grid, y_grid)
    
    z = calc_map(a, mesh_x, mesh_y, edge_func = edge_func, **kargs)
    
    return z

def fit_vessel(image, guess=np.array([15, 0.1, 0.001, 35, 0.1, 0.001]), 
               lbound=[-0.1, -0.01], rbound=[0.1, 0.01]):
    def fit(a, mesh_x1d, mesh_y1d, image1d):
        
        z = calc_map(a, mesh_x1d, mesh_y1d)
        
        return 1-np.abs(pearsonr(z, image1d**0.5)[0])
    
    h, w = image.shape
    x_grid = np.linspace(0, w-1, w)
    y_grid = np.linspace(0, h-1, h)
    mesh_x, mesh_y = np.meshgrid(x_grid, y_grid)
    mesh_x1d = mesh_x.flatten()
    mesh_y1d = mesh_y.flatten()
    image1d = image.flatten()
    a0 = guess
    res_lsq = least_squares(fit, a0, args=(mesh_x1d, mesh_y1d, image1d), bounds=[(5, *lbound, h//2, *lbound), (h//2, *rbound, h, *rbound)])
    return res_lsq

def fit_vessel_fourier(image, guess=np.array([15, 0.1, 0.001, 35, 0.1, 0.001]), 
               lbound=[-np.inf], rbound=[np.inf]):
    def fit(a, mesh_x1d, mesh_y1d, image1d):
        
        z = calc_map(a, mesh_x1d, mesh_y1d, edge_func=calc_edge_fourier)
        
        return 1-np.abs(pearsonr(z, image1d>image1d.mean())[0])
    
    h, w = image.shape
    x_grid = np.linspace(0, w-1, w)
    y_grid = np.linspace(0, h-1, h)
    mesh_x, mesh_y = np.meshgrid(x_grid, y_grid)
    mesh_x1d = mesh_x.flatten()
    mesh_y1d = mesh_y.flatten()
    image1d = image.flatten()
    a0 = guess
    res_lsq = least_squares(fit, a0,method="dogbox",ftol=1e-5, xtol=1e-5, gtol=1e-5, args=(mesh_x1d, mesh_y1d, image1d), bounds=[lbound, rbound])
    return res_lsq
    
def fit(fimage, plot=False, **kwargs):
    if isinstance(fimage, str):
        gimage = cv2.imread(fimage)
    else:
        gimage = fimage
    assert isinstance(gimage, np.ndarray), f"{type(gimage)}"
    gimage = cv2.cvtColor(gimage, cv2.COLOR_BGR2GRAY)
    image = gimage.copy()
    image = image.astype(np.float32)
    image = image**3
    # image = cv2.GaussianBlur(image, (3,3), 1.0)
    # image = (image - image.mean()) / np.std(image)
    # image = image/255.0
    h, w = image.shape
    x_grid = np.linspace(0, w-1, w)
    
    
    if "guess" in kwargs:
        _, init = fourier_init_condition(h)
        guess = kwargs["guess"]
        init = (guess - np.abs(guess)*0.5, guess + np.abs(guess)*0.5)
    else:
        guess, init = fourier_init_condition(h)
        
    result = fit_vessel_fourier(image, guess=guess, lbound=init[0], rbound=init[1])
    # print(result)
    
    edge1, edge2 = calc_edge_fourier(x_grid, result.x)
    
    if plot:
        mask = masking(gimage, result.x, edge_func=calc_edge_fourier)
        figure, axes = plt.subplots(1, 2, figsize=(15, 7))

        axes[0].imshow(gimage, cmap = "gray")
        axes[0].plot(x_grid, edge1, "r-")
        axes[0].plot(x_grid, edge2, "r-")
        axes[1].imshow(mask, cmap = "gray")
        axes[1].plot(x_grid, edge1, "r-")
        axes[1].plot(x_grid, edge2, "r-")

        # plt.imshow(np.concatenate([image, mask], axis=1), cmap = "gray")
        plt.show()
        plt.close()
    
    # image = image*255

    return result.x, x_grid, gimage, edge1, edge2

def animation():
    
    matplotlib.use('Agg')
    
    def init(ax):
        ax.set_xlim(0, 100)
    
    def update(i):
        if i ==0:
            guess, x_grid, image, edge1, edge2 = fit(f"output\\frame{i}.jpg", plot=False)
            thickness = calc_thickness(guess, x_grid)
            xdata.append(i*1/30)
            ydata.append(thickness)
            mask = masking(image, guess, edge_func=calc_edge_fourier)
        else:
            # temp = guess
            guess, x_grid, image, edge1, edge2 = fit(f"output\\frame{i}.jpg", plot=False)
            thickness = calc_thickness(guess, x_grid)
            xdata.append(i*1/30)
            ydata.append(thickness)
            mask = masking(image, guess, edge_func=calc_edge_fourier)
        
        ax[0].clear()
        ax[0].imshow(image, cmap = "gray")
        ax[0].plot(x_grid+0.5, edge1+0.5, "r-")
        ax[0].plot(x_grid+0.5, edge2+0.5, "r-")
        ax[1].clear()
        ax[1].plot(xdata, ydata)
        # ax[1].imshow(mask, cmap = "gray")
        # ax[1].plot(x_grid, edge1, "r-")
        # ax[1].plot(x_grid, edge2, "r-")
        
    fig, ax = plt.subplots(1,2)
    xdata, ydata = [], []
    
    # filelist = glob.glob("output\\*.jpg")
    ani = FuncAnimation(fig, update, frames=np.arange(0, 10000, 5), repeat=False)
    plt.show()
    ani.save('animation.gif', writer='imagemagick', fps=10)

if __name__ == "__main__":
    xdata, ydata = [], []
    # for i in range(1,  1000, 100):
    #     if i ==1:
    #         guess = main(f"output\\frame{i}.jpg")
    #         results.append(guess)
    #     else:
    #         results.append(main(f"output\\frame{i}.jpg", guess = guess))
    
    # animation()
    
    import glob, time, tqdm
    t1 = time.time()
    filelist = glob.glob("output\\*.jpg")
    N = 5
    filelist = filelist[::N]
    dt = 1/(20 / N)
    for i, file in tqdm.tqdm(enumerate(filelist), total=len(filelist)):
        if i ==0:
            guess, x_grid, image, edge1, edge2 = fit(file, plot=True)
            thickness = calc_thickness(guess, x_grid)
            # print(guess)
            xdata.append(i*dt)
            ydata.append(thickness)
        else:
            temp = guess
            guess, x_grid, image, edge1, edge2 = fit(file, plot=True, guess=guess)
            thickness = calc_thickness(guess, x_grid)
            # print(temp - guess)
            xdata.append(i*dt)
            ydata.append(thickness)
    t2 = time.time()
    t = t1-t2
    print(t/60/i)
    
    xdata= np.array(xdata)
    ydata= np.array(ydata)
    plt.figure(figsize=(10, 5))
    plt.plot(xdata, np.convolve(ydata, np.ones(12), 'same') / 12)
    

            
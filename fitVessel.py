import numpy as np
from scipy import special
from scipy.optimize import least_squares
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import cv2

# def pearsonr(x , y):
#     x = np.asarray(x)
#     y = np.asarray(y)
#     n = len(x)
#     mx = x.mean()
#     my = y.mean()
#     xm, ym = x-mx, y-my
#     r_num = np.add.reduce(xm * ym)
#     r_den = np.sqrt(ss(xm) * ss(ym))
#     r = r_num / r_den
#     return r

def calc_edge_fourier(mesh_x, a, N=7):
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

def fourier_init_condition(H, N=7):
    guess = np.array([H*0.25, 0.1, *(0.1,)*2*N, H*0.75, 0.1, *(0.1,)*2*N])
    bounds = (np.array([0, *(-np.inf,)*(2*N+1), H//2, *(-np.inf,)*(2*N+1)]), np.array([H//2, *(np.inf,)*(2*N+1), H, *(np.inf,)*(2*N+1)]))
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
    
    z1 = -special.erf((mesh_y - edge1) * 1)
    z2 = special.erf((mesh_y - edge2) * 1)
    z = z1 + z2
    
    return z

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
        
        return 1-np.abs(pearsonr(z, image1d**0.5)[0])
    
    h, w = image.shape
    x_grid = np.linspace(0, w-1, w)
    y_grid = np.linspace(0, h-1, h)
    mesh_x, mesh_y = np.meshgrid(x_grid, y_grid)
    mesh_x1d = mesh_x.flatten()
    mesh_y1d = mesh_y.flatten()
    image1d = image.flatten()
    a0 = guess
    res_lsq = least_squares(fit, a0, args=(mesh_x1d, mesh_y1d, image1d), bounds=[lbound, rbound])
    return res_lsq
    
def main(fimage, **kwargs):
    
    image = cv2.imread(fimage)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (3,3), 0)
    # image = (image - image.mean()) / np.std(image)
    image = image/255.0
    h, w = image.shape
    x_grid = np.linspace(0, w-1, w)
    
    
    if "guess" in kwargs:
        _, init = fourier_init_condition(h)
        guess = kwargs["guess"]
    else:
        guess, init = fourier_init_condition(h)
        
    result = fit_vessel_fourier(image, guess=guess, lbound=init[0], rbound=init[1])
    # print(result)
    
    mask = masking(image, result.x, edge_func=calc_edge_fourier)
    edge1, edge2 = calc_edge_fourier(x_grid, result.x)
    
    figure, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    axes[0].imshow(image, cmap = "gray")
    axes[0].plot(x_grid, edge1, "r-")
    axes[0].plot(x_grid, edge2, "r-")
    axes[1].imshow(mask, cmap = "gray")
    axes[1].plot(x_grid, edge1, "r-")
    axes[1].plot(x_grid, edge2, "r-")
    
    # plt.imshow(np.concatenate([image, mask], axis=1), cmap = "gray")
    plt.show()
    plt.close()
    
    return result.x

if __name__ == "__main__":
    for i in range(1,  1000, 100):
        if i ==1:
            guess = main(f"output\\frame{i}.jpg")
        else:
            main(f"output\\frame{i}.jpg", guess = guess)
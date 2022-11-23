import os
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import pearsonr
# from matplotlib import pyplot as plt
# import matplotlib
# from matplotlib.animation import FuncAnimation
import cv2
import copy
from skimage.exposure import match_histograms

CANNY_LOWER=90
CANNY_HIGHER=200

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
        edge1 += a[i*2] * np.cos(w1 * i * (mesh_x)) + a[i*2+1] * np.sin(w1 * i *mesh_x)
        edge2 += a[i*2+2*N+2] * np.cos(w2 * i * mesh_x) + a[i*2+3+2*N] * np.sin(w2 * i *mesh_x)
    
    return edge1, edge2

def fourier_init_condition(H, N=5):
    guess = np.array([H*0.25, 0.3, *(0.1,)*2*N, H*0.75, 0.3, *(0.1,)*2*N])
    bounds = (np.array([0, 0.1,*(-np.inf,)*2*N, H//2, 0.2, *(-np.inf,)*2*N]), np.array([H//2, 3, *(np.inf,)*2*N, H, 0.7, *(np.inf,)*2*N]))
    return guess, bounds

def calc_edge_polynomial(mesh_x, a, order=1):
    
    edge1 = a[0]
    edge2 = a[order+1]
    mid = mesh_x.mean()
    
    for i in range(1, order+1):
        edge1 += a[i]*(mesh_x-mid)**i
        edge2 += a[i+order+1]*(mesh_x-mid)**i
    
    return edge1, edge2

def polynomial_init_condition(H, order=1):
    if order==1:
        guess = np.array([H*0.25, 0, H*0.75, 0])
        bounds = (np.array([H*0.05, -0.2, H//2, -0.2]), np.array([H//2,0.2,  H*0.95, 0.2]))
    else:
        guess = np.array([H*0.25, *[0.1**i for i in range(1, order+1)], H*0.75, *[0.1**i for i in range(1, order+1)]])
        bounds = (np.array([0, *(-np.inf,)*order, H//2, *(-np.inf,)*order]), np.array([H/2, *(np.inf,)*order,H, *(np.inf,)*order]))
    return guess, bounds

def calc_map(a, mesh_x, mesh_y, edge_func=calc_edge_polynomial, **kargs):
    edge1, edge2 = edge_func(mesh_x, a, **kargs)
    

    # z1 = -special.erf((mesh_y - edge1) * 1)
    # z2 = special.erf((mesh_y - edge2) * 1)
    
    z1 = -sigmoid((mesh_y - edge1)*2)
    z2 =  sigmoid((mesh_y - edge2)*2)
    # z1 =  sigmoid(np.abs(mesh_y - edge1)*0.5)
    # z2 =  sigmoid(np.abs(mesh_y - edge2)*0.5)
    
    z = z1 + z2
    
    return z

def calc_thickness(a, mesh_x, edge_func=calc_edge_fourier):
    
    edge1, edge2 = edge_func(mesh_x, a)
    
    midline = (edge1 + edge2) / 2
    midline_arc = np.sum(np.sqrt((midline[:-1] - midline[1:])**2 + 1))
    thickness = np.sum(np.abs(edge1 - edge2)) / midline_arc
    
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
        
        z = calc_map(a, mesh_x1d, mesh_y1d, edge_func=calc_edge_polynomial).flatten()
        
        return 1-np.abs(pearsonr(z, image1d)[0])
    
    h, w = image.shape
    x_grid = np.linspace(0, w-1, w)
    y_grid = np.linspace(0, h-1, h)
    mesh_x, mesh_y = np.meshgrid(x_grid, y_grid)
    mesh_x1d = mesh_x.flatten()
    mesh_y1d = mesh_y.flatten()
    image1d = image.flatten()
    a0 = guess
    res_lsq = least_squares(fit, a0, args=(mesh_x1d, mesh_y1d, image1d), bounds=[lbound, rbound], method="trf",ftol=1e-4, xtol=1e-4, gtol=1e-4)
    return res_lsq

def fit_vessel_fourier(image, guess=np.array([15, 0.1, 0.001, 35, 0.1, 0.001]), 
               lbound=[-np.inf], rbound=[np.inf]):
    def fit(a, mesh_x1d, mesh_y1d, image1d):
        
        z = calc_map(a, mesh_x1d, mesh_y1d, edge_func=calc_edge_fourier)
        
        return 1-np.abs(pearsonr(z, image1d)[0])
    
    h, w = image.shape
    x_grid = np.linspace(0, w-1, w)
    y_grid = np.linspace(0, h-1, h)
    mesh_x, mesh_y = np.meshgrid(x_grid, y_grid)
    mesh_x1d = mesh_x.flatten()
    mesh_y1d = mesh_y.flatten()
    image1d = image.flatten()
    a0 = guess
    res_lsq = least_squares(fit, a0, method="trf",ftol=1e-5, xtol=1e-4, gtol=1e-5, args=(mesh_x1d, mesh_y1d, image1d), bounds=[lbound, rbound])
    return res_lsq

def pad_edge(image):
    # nimage = np.where(image==0, 1, 0)
    y , x = np.nonzero(image)
    for i in range(image.shape[1]):
        y_i = y[x==i]
        up = y_i[np.argmin(y_i)]
        down = y_i[np.argmax(y_i)]
        image[0:up, i] = 1
        image[down:, i] = 1
    return image

def fit(fimage, plot=False, init_func=fourier_init_condition, fit_func = fit_vessel_fourier, edge_func=calc_edge_fourier, refimg = None, **kwargs):
    if isinstance(fimage, str):
        gimage = cv2.imread(fimage)
    else:
        gimage = fimage
    assert isinstance(gimage, np.ndarray), f"{type(gimage)}"
    # image = cv2.GaussianBlur(image, (3,3), 1.5)
    if not refimg is None:
        # print(image.shape)
        gimage = match_histograms(gimage, refimg, channel_axis=-1)
        gimage = cv2.cvtColor(gimage, cv2.COLOR_BGR2GRAY)
    else:
        gimage = cv2.cvtColor(gimage, cv2.COLOR_BGR2GRAY)
        gimage = cv2.equalizeHist(gimage)
    bmask = gimage>100
    image = copy.deepcopy(gimage)
    # image = cv2.GaussianBlur(image, (3,3), 2.0)
    
    image = image*bmask
    image = cv2.Canny(image, CANNY_LOWER, CANNY_HIGHER, L2gradient=True)
    image = np.where(cv2.GaussianBlur(image, (3,3), 2.0)>0, 1, 0)
    image = cv2.morphologyEx(image.astype(np.float32), cv2.MORPH_CLOSE, np.ones((5,5)), iterations=1)
    image = pad_edge(image)
    image = cv2.morphologyEx(image.astype(np.float32), cv2.MORPH_CLOSE, np.ones((5,5)), iterations=1)
    image = image.astype(np.float32)
    # image = image/255.0
    # image = image**1
    # image= image.astype(np.uint8)
    # ret,image = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # image = (image - image.mean()) / np.std(image)
    h, w = image.shape
    x_grid = np.linspace(0, w-1, w)
    
    
    guess, init = init_func(h)
    if "guess" in kwargs:
        guess = kwargs["guess"]
        init = (guess - np.abs(guess)*0.15, guess + np.abs(guess)*0.15)
        
    assert guess.shape==init[0].shape and guess.shape == init[1].shape
    result = fit_func(image, guess=guess, lbound=init[0], rbound=init[1])
    # print(result)
    
    edge1, edge2 = edge_func(x_grid, result.x)
    
    if plot:
        mask = masking(gimage, result.x, edge_func=edge_func)
        figure, axes = plt.subplots(1, 3, figsize=(15, 7))

        axes[0].imshow(gimage, cmap = "gray")
        axes[0].plot(x_grid, edge1, "r-")
        axes[0].plot(x_grid, edge2, "r-")        
        axes[1].imshow(image, cmap = "gray")
        axes[1].plot(x_grid, edge1, "r-")
        axes[1].plot(x_grid, edge2, "r-")
        axes[2].imshow(mask, cmap = "gray")
        axes[2].plot(x_grid, edge1, "r-")
        axes[2].plot(x_grid, edge2, "r-")

        # plt.imshow(np.concatenate([image, mask], axis=1), cmap = "gray")
        if isinstance(fimage, str):
            plt.title(os.path.basename(fimage))
        if "to_file" in kwargs:
            figure.savefig(kwargs["to_file"], dpi=150)
        plt.show()
        plt.close()
    
    # image = image*255

    return result.x, x_grid, gimage, edge1, edge2
'''
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
'''
if __name__ == "__main__":
    from matplotlib import pyplot as plt
    xdata, ydata = [], []
    # for i in range(1,  1000, 100):
    #     if i ==1:
    #         guess = main(f"output\\frame{i}.jpg")
    #         results.append(guess)
    #     else:
    #         results.append(main(f"output\\frame{i}.jpg", guess = guess))
    
    # animation()
    refimg = cv2.imread(r"ref.png")
    # refimg = cv2.cvtColor(refimg, cv2.COLOR_BGR2GRAY)
    
    import glob, time
    t1 = time.time()
    filelist = glob.glob(r"C:\Users\wangs\dev\Ultrasond\test\orgCropped\*.png")
    N = 5
    # filelist = filelist[::N]
    dt = 1/(20 / N)
    for i, file in enumerate(filelist):
        if i ==0:
            img = cv2.imread(file)
            guess, x_grid, image, edge1, edge2 = fit(img, plot=True, init_func=polynomial_init_condition, edge_func=calc_edge_polynomial, fit_func=fit_vessel, refimg=refimg)
            temp = guess
            gimage = copy.deepcopy(image)
        else:
            img = cv2.imread(file)
            guess, x_grid, image, edge1, edge2 = fit(img, plot=True, init_func=polynomial_init_condition, edge_func=calc_edge_polynomial, fit_func=fit_vessel, refimg=refimg)
        thickness = calc_thickness(guess, x_grid, edge_func=calc_edge_polynomial)
        # print(temp - guess)
        xdata.append(i*dt)
        ydata.append(thickness)
    t2 = time.time()
    t = t1-t2
    print(t/60/i)
    
    xdata= np.array(xdata)
    ydata= np.array(ydata)

    from scipy import signal
    b, a = signal.butter(2, 1/(23), "low", analog = False) 
    col_filtered = signal.filtfilt(b, a, ydata)
    
    plt.figure(figsize=(10, 5))
    plt.plot(xdata, ydata)
    # plt.plot(xdata, np.convolve(ydata, np.ones(12), 'same') / 12)
    plt.plot(xdata, col_filtered)
    plt.title(f"{CANNY_LOWER}-{CANNY_HIGHER}")
    

            
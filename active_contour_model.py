import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.io import imread
from skimage.color import rgb2gray
import scipy.linalg
from scipy.interpolate import RectBivariateSpline
from skimage.draw import circle_perimeter
from skimage.filters import gaussian_filter, sobel, scharr
from skimage import data
import warnings

def _zero_outside_range(f, maxx, maxy):
    def func(x, y, **kwargs):
        return f(x, y, **kwargs)*\
            np.logical_and(x<=maxx, y<=maxy)
    return func

def active_contour_model(image, snake, alpha=0.01, beta=0.1,
                         w_line=0, w_edge=1, gamma=0.01,
                         bc='periodic', max_iterations=2500,
                         convergence=0.25):
    """Active contour model

    Active contours by fitting snakes to features of images. Supports single
    and multichannel 2D images. Snakes can be periodic (for segmentation) or
    have fixed and/or free ends.

    Parameters
    ----------
    image: (N, M) or (N, M, 3) ndarray
        Input image
    snake: (N, 2) ndarray
        Initialisation of snake.
    alpha: float, optional
        Snake length shape parameter
    beta: float, optional
        Snake smoothness shape parameter
    w_line: float, optional
        Controls attraction to brightness. Use negative values to attract to
        dark regions
    w_edge: float, optional
        Controls attraction to edges. Use negative values to repel snake from
        edges.
    gamma: flota, optional
        Excpliti time stepping parameter.
    bc: {'periodic', 'free', 'fixed'}, optional
        Boundary conditions for worm. 'periodic' attaches the two ends of the
         snake, 'fixed' holds the end-points in place, and'free' allows free
         movement of the ends. 'fixed' and 'free' can be combined by parsing
         'fixed-free', 'free-fixed'. Parsing 'fixed-fixed' or 'free-free'
         yields same behaviour as 'fixed' and 'free', respectively.
    max_iterations: int, optional
        Maximum iterations to optimize snake shape.
    convergence: float, optional
        Convergence criteria.

    Returns
    -------
    snake: (N, 2) ndarray
        Optimised snake, same shape as input parameter.

    References
    ----------
    .. [1]  Kass, M.; Witkin, A.; Terzopoulos, D. "Snakes: Active contour models". International Journal of Computer Vision 1 (4): 321 (1988).

    Examples
    --------

    """
    convergence_order = 10
    valid_bcs = ['periodic', 'free', 'fixed', 'free-fixed',
                 'fixed-free', 'fixed-fixed', 'free-free']
    if bc not in valid_bcs:
        raise ValueError("Invalid boundary condition.\n"+
                         "Should be one of: "+", ".join(valid_bcs)+'.')
    img = img_as_float(image)
    RGB = len(img.shape)==3
    if RGB:
        edge = [sobel(img[:,:,0]),sobel(img[:,:,1]),sobel(img[:,:,2])]
    else:
        edge = [sobel(img)]
    for i in xrange(3 if RGB else 1):
        edge[i][0,:] = edge[i][1,:]
        edge[i][-1,:] = edge[i][-2,:]
        edge[i][:,0] = edge[i][:,1]
        edge[i][:,-1] = edge[i][:,-2]
    if RGB:
        img = w_line*np.sum(img,axis=2) \
            + w_edge*sum(edge)
    else:
        img = w_line*img + w_edge*edge[0]

    intp = RectBivariateSpline(np.arange(img.shape[1]),
            np.arange(img.shape[0]), img.T, kx=2, ky=2, s=0)
    intp = _zero_outside_range(intp, img.shape[1], img.shape[0])

    x, y = snake[:, 0].copy(), snake[:, 1].copy()
    xsave = np.empty((convergence_order,len(x)))
    ysave = np.empty((convergence_order,len(x)))

    n = len(x)
    a = np.roll(np.eye(n), -1, axis=0) \
      + np.roll(np.eye(n), -1, axis=1) \
      - 2*np.eye(n)
    b = np.roll(np.eye(n), -2, axis=0) \
      + np.roll(np.eye(n), -2, axis=1) \
      - 4*np.roll(np.eye(n), -1, axis=0) \
      - 4*np.roll(np.eye(n), -1, axis=1) \
      + 6*np.eye(n)
    A = -alpha*a + beta*b
    sfixed = False
    if bc.startswith('fixed'):
        A[0, :] = 0
        A[1, :] = 0
        A[1, :3] = [1, -2, 1]
        sfixed = True
    efixed = False
    if bc.endswith('fixed'):
        A[-1, :] = 0
        A[-2, :] = 0
        A[-2, -3:] = [1, -2, 1]
        efixed = True
    sfree = False
    if bc.startswith('free'):
        A[0, :] = 0
        A[0, :3] = [1, -2, 1]
        A[1, :] = 0
        A[1, :4] = [-1, 3, -3, 1]
        sfree = True
    efree = False
    if bc.endswith('free'):
        A[-1, :] = 0
        A[-1, -3:] = [1, -2, 1]
        A[-2, :] = 0
        A[-2, -4:] = [-1, 3, -3, 1]
        efree = True

    inv = scipy.linalg.inv(A+gamma*np.eye(n))
    for i in xrange(max_iterations):
        fx = intp(x, y, dx=1, grid=False)
        fy = intp(x, y, dy=1, grid=False)
        if sfixed:
            fx[0] = 0
            fy[0] = 0
        if efixed:
            fx[-1] = 0
            fy[-1] = 0
        if sfree:
            fx[0] *= 2
            fy[0] *= 2
        if efree:
            fx[-1] *= 2
            fy[-1] *= 2
        xn = np.dot(inv, gamma*x + fx)
        yn = np.dot(inv, gamma*y + fy)
        x[:] += np.tanh(xn-x)
        y[:] += np.tanh(yn-y)

        # Convergence criteria:
        j = i%(convergence_order+1)
        if j<convergence_order:
            xsave[j,:] = x
            ysave[j,:] = y
        else:
            dist = np.min(np.max(np.abs(xsave-x[None, :])
                + np.abs(ysave-y[None, :]), 1))
            if dist < convergence:
                break

    return np.array([x, y]).T

if __name__ == '__main__':
    img = data.astronaut()
    img = rgb2gray(img)

    s = np.linspace(0,2*np.pi,400)
    x = 220 + 100*np.cos(s)
    y = 100 + 100*np.sin(s)
    init = np.array([x, y]).T
    snake = active_contour_model(gaussian_filter(img,3),
        init, alpha=0.015, beta=10, w_line=0, w_edge=1, gamma=0.001)

    plt.clf()
    plt.gray()
    plt.imshow(img)
    plt.plot(init[:,0],init[:,1],'--r')
    plt.plot(snake[:,0],snake[:,1],'-b')
    plt.axis([0, img.shape[1], img.shape[0], 0])
    plt.show()

    img = data.text()
    img = rgb2gray(img)

    x = np.linspace(5,424,100)
    y = np.linspace(136,50,100)
    init = np.array([x, y]).T

    snake = active_contour_model(gaussian_filter(img,1), init, bc='fixed',
            alpha=0.1, beta=1.0, w_line=-5, w_edge=0, gamma=0.1)
    plt.clf()
    plt.gray()
    plt.imshow(img)
    plt.plot(init[:,0],init[:,1],'--r')
    plt.plot(snake[:,0],snake[:,1],'-b')
    plt.axis([0, img.shape[1], img.shape[0], 0])
    plt.show()

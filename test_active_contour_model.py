import numpy as np
from skimage import data
from skimage.color import rgb2gray
from skimage.filters import gaussian_filter, sobel
from active_contour_model import active_contour_model
from numpy.testing import assert_equal, assert_allclose, assert_raises

def periodic_reference_test():
    img = data.astronaut()
    img = rgb2gray(img)
    s = np.linspace(0,2*np.pi,400)
    x = 220 + 100*np.cos(s)
    y = 100 + 100*np.sin(s)
    init = np.array([x, y]).T
    snake = active_contour_model(gaussian_filter(img,3), init,
        alpha=0.015, beta=10, w_line=0, w_edge=1, gamma=0.001)
    refx = [299, 298, 298, 298, 298, 297, 297, 296, 296, 295]
    refy = [98, 99, 100, 101, 102, 103, 104, 105, 106, 108]
    assert_equal(np.array(snake[:10,0], dtype=np.int32), refx)
    assert_equal(np.array(snake[:10,1], dtype=np.int32), refy)


def fixed_reference_test():
    img = data.text()
    x = np.linspace(5,424,100)
    y = np.linspace(136,50,100)
    init = np.array([x, y]).T
    snake = active_contour_model(gaussian_filter(img,1), init, bc='fixed',
            alpha=0.1, beta=1.0, w_line=-5, w_edge=0, gamma=0.1)
    refx = [5, 9, 13, 17, 21, 25, 30, 34, 38, 42]
    refy = [136, 135, 134, 133, 132, 131, 129, 128, 127, 125]
    assert_equal(np.array(snake[:10,0], dtype=np.int32), refx)
    assert_equal(np.array(snake[:10,1], dtype=np.int32), refy)


def free_reference_test():
    img = data.text()
    x = np.linspace(5,424,100)
    y = np.linspace(70,40,100)
    init = np.array([x, y]).T
    snake = active_contour_model(gaussian_filter(img,3), init, bc='free',
            alpha=0.1, beta=1.0, w_line=-5, w_edge=0, gamma=0.1)
    refx = [10, 13, 16, 19, 23, 26, 29, 32, 36, 39]
    refy = [76, 76, 75, 74, 73, 72, 71, 70, 69, 69]
    assert_equal(np.array(snake[:10,0], dtype=np.int32), refx)
    assert_equal(np.array(snake[:10,1], dtype=np.int32), refy)


def RGB_test():
    img = gaussian_filter(data.text(),1)
    imgR = np.zeros((img.shape[0],img.shape[1],3))
    imgG = np.zeros((img.shape[0],img.shape[1],3))
    imgRGB = np.zeros((img.shape[0],img.shape[1],3))
    imgR[:,:,0] = img
    imgG[:,:,1] = img
    imgRGB[:,:,:] = img[:, :, None]
    x = np.linspace(5,424,100)
    y = np.linspace(136,50,100)
    init = np.array([x, y]).T
    snake = active_contour_model(imgR, init, bc='fixed',
            alpha=0.1, beta=1.0, w_line=-5, w_edge=0, gamma=0.1)
    refx = [5, 9, 13, 17, 21, 25, 30, 34, 38, 42]
    refy = [136, 135, 134, 133, 132, 131, 129, 128, 127, 125]
    assert_equal(np.array(snake[:10,0], dtype=np.int32), refx)
    assert_equal(np.array(snake[:10,1], dtype=np.int32), refy)
    snake = active_contour_model(imgG, init, bc='fixed',
            alpha=0.1, beta=1.0, w_line=-5, w_edge=0, gamma=0.1)
    assert_equal(np.array(snake[:10,0], dtype=np.int32), refx)
    assert_equal(np.array(snake[:10,1], dtype=np.int32), refy)
    snake = active_contour_model(imgRGB, init, bc='fixed',
            alpha=0.1, beta=1.0, w_line=-5/3., w_edge=0, gamma=0.1)
    assert_equal(np.array(snake[:10,0], dtype=np.int32), refx)
    assert_equal(np.array(snake[:10,1], dtype=np.int32), refy)


def end_points_tests():
    img = data.astronaut()
    img = rgb2gray(img)
    s = np.linspace(0,2*np.pi,400)
    x = 220 + 100*np.cos(s)
    y = 100 + 100*np.sin(s)
    init = np.array([x, y]).T
    snake = active_contour_model(gaussian_filter(img,3), init,
        bc='periodic', alpha=0.015, beta=10, w_line=0, w_edge=1, gamma=0.001,
        max_iterations=100)
    assert np.sum(np.abs(snake[0,:]-snake[-1,:]) ) < 2
    snake = active_contour_model(gaussian_filter(img,3), init,
        bc='free', alpha=0.015, beta=10, w_line=0, w_edge=1, gamma=0.001,
        max_iterations=100)
    assert np.sum(np.abs(snake[0,:]-snake[-1,:])) > 2
    snake = active_contour_model(gaussian_filter(img,3), init,
        bc='fixed', alpha=0.015, beta=10, w_line=0, w_edge=1, gamma=0.001,
        max_iterations=100)
    assert_allclose(snake[0,:], [x[0], y[0]], atol=1e-5)


def bad_input_tests():
    img = np.zeros((10, 10))
    x = np.linspace(5, 424, 100)
    y = np.linspace(136, 50, 100)
    init = np.array([x, y]).T
    np.testing.assert_raises(ValueError, active_contour_model, img, init,
                            bc='wrong')
    np.testing.assert_raises(ValueError, active_contour_model, img, init,
                            max_iterations=-15)


if __name__ == "__main__":
    np.testing.run_module_suite()

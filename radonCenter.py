import numpy as np
from scipy.interpolate import interp2d

def samplingRegion(size_window, theta = [45, 135], m = 0.2, M = 0.8, step = 1, decimals = 2):
    """This function returns all the coordinates of the sampling region, the center of the region is (0,0)
    When applying to matrices, don't forget to SHIFT THE CENTER!
    Input:
        size_window: the radius of the sampling region. The whole region should thus have a length of 2*size_window+1.
        theta: the angle range of the sampling region, default: [45, 135] for the anti-diagonal and diagonal directions.
        m: the minimum fraction of size_window, default: 0.2 (i.e., 20%). In this way, the saturated region can be excluded.
        M: the maximum fraction of size_window, default: 0.8 (i.e., 80%). Just in case if there's some star along the diagonals.
        step: the seperation between sampling dots (units: pixel), default value is 1pix.
        decimals: the precisoin of the sampling dots (units: pixel), default value is 0.01pix.
    Output: (xs, ys)
        xs: x indecies, flattend.
        ys: y indecies, flattend.
    Example:
        If you call "xs, ys = samplingRegion(5)", you will get:
        xs: array([-2.83, -2.12, -1.41, -0.71,  0.71,  1.41,  2.12,  2.83,  2.83, 2.12,  1.41,  0.71, -0.71, -1.41, -2.12, -2.83]
        ys: array([-2.83, -2.12, -1.41, -0.71,  0.71,  1.41,  2.12,  2.83, -2.83, -2.12, -1.41, -0.71,  0.71,  1.41,  2.12,  2.83]))
    """
    theta = np.array(theta)
    zeroDegXs = np.append(np.arange(-int(size_window*M), -int(size_window*m) + 0.1 * step, step), np.arange(int(size_window*m), int(size_window*M) + 0.1 * step, step))
    #create the x indecies for theta = 0, will be used in the loop.
    zeroDegYs = np.zeros(zeroDegXs.size)
    
    xs = np.zeros((np.size(theta), np.size(zeroDegXs)))
    ys = np.zeros((np.size(theta), np.size(zeroDegXs)))
    
    for i, angle in enumerate(theta):
        degRad = np.deg2rad(angle)
        angleDegXs = np.round(zeroDegXs * np.cos(degRad), decimals = decimals)
        angleDegYs = np.round(zeroDegXs * np.sin(degRad), decimals = decimals)
        xs[i, ] = angleDegXs
        ys[i, ] = angleDegYs
    
    xs = xs.flatten()
    ys = ys.flatten()

    return xs, ys


def smoothCostFunction(costFunction, halfWidth = 0):
    """
    smoothCostFunction will smooth the function within +/- halfWidth, i.e., to replace the value with the average within +/- halfWidth pixel.
    Input:
        costFunction: original cost function, a matrix.
        halfWdith: the half width of the smoothing region, default = 0 pix.
    Output:
        newFunction: smoothed cost function.
    """
    if halfWidth == 0:
        return costFunction
    else:
        newFunction = np.zeros(costFunction.shape)
        rowRange = np.arange(costFunction.shape[0], dtype=int)
        colRange = np.arange(costFunction.shape[1], dtype=int)
        rangeShift = np.arange(-halfWidth, halfWidth + 0.1, dtype=int)
        for i in rowRange:
            for j in colRange:
                surrondingNumber = (2 * halfWidth + 1) ** 2
                avg = 0
                for ii in (i + rangeShift):
                    for jj in (j + rangeShift):
                        if (not (ii in rowRange)) or (not (jj in colRange)):
                            surrondingNumber -= 1
                        else:
                            avg += costFunction[ii, jj]
                newFunction[i, j] = avg * 1.0 / surrondingNumber
    return newFunction
    
 
def searchCenter(image, x_ctr_assign, y_ctr_assign, size_window, m = 0.2, M = 0.8, size_cost = 5, theta = [45, 135], smooth = 2, decimals = 2):
    """
    This function searches the center in a grid, 
    calculate the cost function of Radon Transform (Pueyo et al., 2015), 
    then interpolate the cost function, 
    get the center which corresponds to the maximum value in the cost function.
    
    Input:
        image: 2d array.
        x_ctr_assign: the assigned x-center, or starting x-position; for STIS, the "CRPIX1" header is suggested.
        x_ctr_assign: the assigned y-center, or starting y-position; for STIS, the "CRPIX2" header is suggested.
        size_window: half width of the sampling region; size_window = image.shape[0]/2 is suggested.
            m & M:  The sampling region will be (-M*size_window, -m*size_window)U(m*size_window, M*size_window).
        size_cost: search the center within +/- size_cost pixels, i.e., a square region.
        theta: the angle range of the sampling region; default: [45, 135] for the anti-diagonal and diagonal directions.
        smooth: smooth the cost function, for one pixel, replace it by the average of its +/- smooth neighbours; defualt = 2.
        decimals: the precision of the centers; default = 2 for a precision of 0.01.
    Output:
        x_cen, y_cen
    """
    (y_len, x_len) = image.shape

    x_range = np.arange(x_len)
    y_range = np.arange(y_len)

    image_interp = interp2d(x_range, y_range, image, kind = 'cubic')
    #interpolate the image
    
    
    precision = 1
    x_centers = np.round(np.arange(x_ctr_assign - size_cost, x_ctr_assign + size_cost + precision/10.0, precision), decimals=1)
    y_centers = np.round(np.arange(y_ctr_assign - size_cost, y_ctr_assign + size_cost + precision/10.0, precision), decimals=1)
    costFunction = np.zeros((x_centers.shape[0], y_centers.shape[0]))
    #The above 3 lines create the centers of the search region
    #The cost function stores the sum of all the values in the sampling region
    
    size_window = size_window - size_cost
    (xs, ys) = samplingRegion(size_window, theta, m = m, M = M)
    #the center of the sampling region is (0,0), don't forget to shift the center!

    for j, x0 in enumerate(x_centers):
        for i, y0 in enumerate(y_centers):
            value = 0
            
            for x1, y1 in zip(xs, ys):
                x = x0 + x1    #Shifting the center, this now is the coordinate of the RAW IMAGE
                y = y0 + y1
            
                value += image_interp(x, y)
        
            costFunction[i, j] = value  #Create the cost function

    costFunction = smoothCostFunction(costFunction, halfWidth = smooth)
    #Smooth the cost function
    
    interp_costfunction = interp2d(x_centers, y_centers, costFunction, kind='cubic')
    
    
    for decimal in range(1, decimals+1):
        precision = 10**(-decimal)
        if decimal >= 2:
            size_cost = 10*precision
        x_centers_new = np.round(np.arange(x_ctr_assign - size_cost, x_ctr_assign + size_cost + precision/10.0, precision), decimals=decimal)
        y_centers_new = np.round(np.arange(y_ctr_assign - size_cost, y_ctr_assign + size_cost + precision/10.0, precision), decimals=decimal)
    
        x_cen = 0
        y_cen = 0
        maxcostfunction = 0
    
        for x in x_centers_new:
            for y in y_centers_new:
                value = interp_costfunction(x, y)
                if value >= maxcostfunction:
                    maxcostfunction = value
                    x_cen = x
                    y_cen = y
        
        x_ctr_assign = x_cen
        y_ctr_assign = y_cen    
       
    x_cen = round(x_cen, decimals)
    y_cen = round(y_cen, decimals)
    return x_cen, y_cen

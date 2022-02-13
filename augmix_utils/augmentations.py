from numba import njit
import numpy as np
import math

@njit
def enhance_contrast(image_matrix, bins=256):
    image_flattened = image_matrix.flatten()
    image_hist = np.zeros(bins)

    # frequency count of each pixel
    for pix in image_matrix:
        image_hist[pix] += 1

    # cummulative sum
    cum_sum = np.cumsum(image_hist)
    norm = (cum_sum - cum_sum.min()) * 255
    # normalization of the pixel values
    n_ = cum_sum.max() - cum_sum.min()
    uniform_norm = norm / n_
    uniform_norm = uniform_norm.astype('int')

    # flat histogram
    image_eq = uniform_norm[image_flattened]
    # reshaping the flattened matrix to its original shape
    image_eq = np.reshape(a=image_eq, newshape=image_matrix.shape)

    return image_eq

@njit
def equalize_this(image):
    image_src = image

    r_image = image_src[:, :, 0]
    g_image = image_src[:, :, 1]
    b_image = image_src[:, :, 2]

    r_image_eq = enhance_contrast(image_matrix=r_image)
    g_image_eq = enhance_contrast(image_matrix=g_image)
    b_image_eq = enhance_contrast(image_matrix=b_image)

    image_eq = np.dstack(tup=(r_image_eq, g_image_eq, b_image_eq))
    cmap_val = None
    return image_eq

@njit
def solarize(image):
    thresh_val = 130
    image_src = image

    r_image, g_image, b_image = image_src[:, :, 0], image_src[:, :, 1], image_src[:, :, 2]
    ## inverting the colored image (partially)
    r_sol = np.where((r_image < thresh_val), r_image, ~r_image)
    g_sol = np.where((g_image < thresh_val), g_image, ~g_image)
    b_sol = np.where((b_image < thresh_val), b_image, ~b_image)
    image_sol = np.dstack(tup=(r_sol, g_sol, b_sol))

    return image_sol 

@njit
def shear_helper(angle,x,y):

    tangent=math.tan(angle/2)
    new_x=round(x-y*tangent)
    new_y=y

    new_y=round(new_x*math.sin(angle)+new_y)

    new_x=round(new_x-new_y*tangent)
    
    return new_y,new_x

@njit
def shear(image):

    angle = np.random.randint(360)
    # Define the most occuring variables
    angle=math.radians(angle)                               #converting degrees to radians
    cosine=math.cos(angle)
    sine=math.sin(angle)

    height=image.shape[0]                                   #define the height of the image
    width=image.shape[1]                                    #define the width of the image

    # Define the height and width of the new image that is to be formed
    new_height  = round(abs(image.shape[0]*cosine)+abs(image.shape[1]*sine))+1
    new_width  = round(abs(image.shape[1]*cosine)+abs(image.shape[0]*sine))+1

    # define another image variable of dimensions of new_height and new _column filled with zeros
    output=np.zeros((new_height,new_width,image.shape[2]))
    image_copy=output.copy()


    # Find the centre of the image about which we have to rotate the image
    original_centre_height   = round(((image.shape[0]+1)/2)-1)    #with respect to the original image
    original_centre_width    = round(((image.shape[1]+1)/2)-1)    #with respect to the original image

    # Find the centre of the new image that will be obtained
    new_centre_height= round(((new_height+1)/2)-1)        #with respect to the new image
    new_centre_width= round(((new_width+1)/2)-1)          #with respect to the new image


    for i in range(height):
        for j in range(width):
            #co-ordinates of pixel with respect to the centre of original image
            y=image.shape[0]-1-i-original_centre_height                   
            x=image.shape[1]-1-j-original_centre_width 

            #Applying shear Transformation                     
            new_y,new_x=shear_helper(angle,x,y)

            '''since image will be rotated the centre will change too, 
                so to adust to that we will need to change new_x and new_y with respect to the new centre'''
            new_y=new_centre_height-new_y
            new_x=new_centre_width-new_x
            
            output[new_y,new_x,:]=image[i,j,:]
    return output

@njit 
def augment(image):

    num_augmentations = 3
    aug_fn = np.random.choice(num_augmentations)
    if aug_fn == 0:
        continue
    elif aug_fn == 1:
        continue
    elif aug+fn == 2:
        continue
    
    return image

@njit
def aug(image, severity=3, width=3, depth=-1, alpha=1.):
 
    ws = np.float32(np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(alpha, alpha))

    mix = np.zeros_like(image)
    for i in range(width):
    image_aug = image.copy()
    d = depth if depth > 0 else np.random.randint(1, 4)
    for _ in range(d):
    #   op = np.random.choice(augmentations.augmentations)
    #   image_aug = apply_op(image_aug, op, severity)
        image_aug = augment(image_aug)
    # Preprocessing commutes since all coefficients are convex
    mix += ws[i] * normalize(image_aug)

    mixed = (1 - m) * normalize(image) + m * mix
    return mixed
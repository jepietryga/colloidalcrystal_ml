import numpy as np
import copy
import cv2
from skimage.filters import threshold_local
from functools import partial
from skimage import feature, future

def neighborhood_maxima(img,grid_spacing):
    '''
    Intended to be used for getting localized maxima, most important for modular distance transform
    '''
    (y_max,x_max) = np.shape(img)
    x_spacing = np.round(np.linspace(0,x_max,grid_spacing))
    y_spacing = np.round(np.linspace(0,y_max,grid_spacing))

    print(x_spacing,y_spacing)
    max_arr = []
    for ii in range(grid_spacing-1):
        for jj in range(grid_spacing-1):
            tb = int(y_spacing[jj])
            bb = int(y_spacing[jj+1])
            lb = int(x_spacing[ii])
            rb = int(x_spacing[ii+1])
            max_arr.append(np.max(
                    img[tb:bb,lb:rb])
                )
    
    return max_arr

def get_padded_stack(img,kernel_dim):
    '''
    Helper function to stack for along z-axis operations
    '''
    pad_width = int(np.floor(kernel_dim/2))
    kernel_mid = np.ceil(kernel_dim/2)
    padded_img = np.pad(img,pad_width=pad_width,constant_values=np.inf)

    # Define roll instructions
    roll_rule = np.atleast_2d(np.array(range(kernel_dim))+1-kernel_mid)
    x_rule = np.concatenate([roll_rule]*kernel_dim,axis=0).astype(int)
    y_rule = x_rule.T
    total_rule = list(zip(np.ravel(y_rule),np.ravel(x_rule)))

    img_stack_list = []
    for shifts in total_rule:
        img_roll = np.roll(padded_img,shift=shifts)
        img_stack_list.append(np.roll(padded_img,shift=shifts))
    
    img_stack = np.stack(img_stack_list,axis=0)

    img_ret = img_stack[:,pad_width:pad_width*-1,pad_width:pad_width*-1]
    return img_ret

def quick_close(img,kernel,neighbor_threshold=4):
    '''
    Given a binary image, see which "holes" to clsoe based on the number of hole neighbors
    '''
    img_ret = get_padded_stack(img,kernel)

    original_logical = img == 0
    stack_logical = np.sum(img_ret,axis=0) <= ((kernel**2)-(1+neighbor_threshold)) #+1 acccounts for self, <= because holes

    img_close = copy.deepcopy(img)
    img_close[original_logical & ~stack_logical] = 1
    return img_close

def kernel_minima(img,kernel_dim:int):
    img_ret = np.min(get_padded_stack(img,kernel_dim),axis=0)
    assert(np.shape(img_ret) == np.shape(img))
    return img_ret

def kernel_maxima(img,kernel_dim:int):
    img_ret = np.max(get_padded_stack(img,kernel_dim),axis=0)
    assert(np.shape(img_ret) == np.shape(img))
    return img_ret

def kernel_range(img,kernel_dim:int):
    img_ret = kernel_maxima(img,kernel_dim)-kernel_minima(img,kernel_dim)
    return img_ret

def z_wrap_2d_kernel(kernel_array):
    return np.reshape(kernel_array,(len(kernel_array),1,1))

def normalized_kernel_stack(img,kernel_dim:int):
    '''
    Create an image stack, then normalize along the z direction
    '''
    img_ret = get_padded_stack(img,kernel_dim)

    img_ret = img_ret/np.max(img_ret,axis=0)

    return img_ret

def normalized_neighbor_sum(img,kernel_dim):
    return np.sum(normalized_kernel_stack(img,kernel_dim),axis=0)

def canny_edge_cv2(
        image,
        blur_size=None,
        tl=None,
        tu=None,
        d=None,
        ss=None,
        use_bilateral=False,
    ):

        if not use_bilateral:
            img_blur = cv2.GaussianBlur(image, blur_size, 0)
            # img_blur = cv2.GaussianBlur(img_blur,blur_size,0)
        else:
            img_blur = cv2.bilateralFilter(image, d, ss, ss)
        edge = cv2.Canny(img_blur, tl, tu, apertureSize=3, L2gradient=True)
        return edge

### Edge Modification Functions, deifne them to return the EDGES, not modify the threshold yet

def edge_canny(self,):
    canny_args = [(5,5),60,120,80,80,False]
    canny_edge = canny_edge_cv2(self.image,*canny_args)

    self._edge_highlight = canny_edge
    return canny_edge

def edge_variance(self):
    # Get Canny Edge
    canny_args = [(5,5),60,120,80,80,False]
    canny_edge = canny_edge_cv2(self.image,*canny_args)

    # Make averaging kernel
    kern_size = 7
    n = 1/kern_size**2
    row = [n for ii in range(kern_size)]
    kernel_list = [row for ii in range(kern_size)]
    kernel = np.array(kernel_list
                    )
    image_sq = self.image**2
    
    # Get variance
    img_edges = cv2.filter2D(image_sq,-1,kernel)-cv2.filter2D(self.image,-1,kernel)**2
    img_edges[canny_edge == 0] = 0

    # Make histograms for deadness/liveness heuristic
    histogram, bin_edges = np.histogram(img_edges,bins=256)

    # remove 0 vals, 255 vals
    
    bin_edges = bin_edges[1:-1]
    histogram = histogram[1:-1]
    cut_off = bin_edges[np.argmax(histogram)]*1

    # Define dead edges
    dead_edges = copy.deepcopy(img_edges)
    dead_edges[dead_edges < cut_off] = 0
    dead_edges[dead_edges > 0] = 255

    # define live edges
    live_edges = copy.deepcopy(img_edges)
    live_edges[live_edges >= cut_off] = 0
    live_edges[live_edges > 0] = 255

    # broaden edges for visibility, store for figure reference
    dead_broad = cv2.GaussianBlur(dead_edges,(3,3),cv2.BORDER_DEFAULT)
    live_broad  = cv2.GaussianBlur(live_edges,(3,3),cv2.BORDER_DEFAULT)
    color_img = cv2.cvtColor(self.image,cv2.COLOR_GRAY2RGB)
    color_img[dead_broad > 0] = (0,0,255)
    color_img[live_broad > 0] = (0,255,0)
    #color_img[(dead_broad > 0) & (live_broad>0)] = (255,0,0)
    self._edge_highlight = color_img
    self._live_edges = live_edges

    # Modify threshold
    self.thresh = self.thresh-live_edges

def edge_darkbright(self):
    # Get Canny Edge
    canny_args = [(5,5),20,50,80,80,False]
    canny_edge = canny_edge_cv2(self.image,*canny_args)

    # Define sharpen kernel
    n = -1
    m = 5
    sharpen_kernel = np.array([[0,n,0],
                    [n,m,n],
                    [0,n,0]]
                    )*1
    
    kern_size = 3
    n = 1/kern_size**2
    row = [n for ii in range(kern_size)]
    kernel_list = [row for ii in range(kern_size)]
    avg_kernel = np.array(kernel_list)
    
    #kernel = np.ones([7,7])*n
    #kernel[3,3] = 49

    # Get just edge intensities
    #img_edges = cv2.filter2D(self.image,-1,sharpen_kernel)
    #canny_edge = cv2.Canny(img_edges,50,100)
    img_edges = cv2.filter2D(self.image,-1,avg_kernel)
    img_edges = cv2.GaussianBlur(self.image, (5,5),0)
    #img_edges = kernel_range(img_edges,5).astype(np.uint8)

    img_edges[canny_edge == 0] = 0

    # Make histograms for deadness/liveness heuristic
    histogram, bin_edges = np.histogram(img_edges,bins=256) #256
    # remove 0 vals, 255 vals
    bin_edges = bin_edges[1:-1]
    histogram = histogram[1:-1]
    mode = bin_edges[np.argmax(histogram)]
    median = np.median(img_edges[img_edges != 0])
    mean = np.mean(img_edges[img_edges != 0])
    self._edge_stats = f"(MED.,MODE,MEAN),({median},{mode},{mean})"
    cut_off = np.mean([median,mean,mode])*1.2 #bin_edges[np.argmax(histogram)]

    # Define dead edges
    dead_edges = copy.deepcopy(img_edges)
    dead_edges[dead_edges < cut_off] = 0
    dead_edges[dead_edges > 0] = 255

    # define live edges
    live_edges = copy.deepcopy(img_edges)
    live_edges[live_edges > cut_off] = 0
    live_edges[live_edges > 0] = 255

    # broaden edges for visibility, store for figure reference
    dead_broad = cv2.GaussianBlur(dead_edges,(3,3),cv2.BORDER_DEFAULT)
    live_broad  = cv2.GaussianBlur(live_edges,(3,3),cv2.BORDER_DEFAULT)
    color_img = cv2.cvtColor(self.image,cv2.COLOR_GRAY2RGB)
    color_img[dead_broad > 0] = (0,0,255)
    color_img[live_broad > 0] = (0,255,0)
    #color_img[(dead_broad > 0) & (live_broad>0)] = (255,0,0)
    self._edge_highlight = color_img
    self._dead_edges = dead_edges
    self._live_edges = live_edges

    return live_edges

def _edge_pixel_classifier(self):
        """
        Experimental:
        Given a RandomForest pixel classifier, identify which edges are valid overlap or facet

        """
        # Load model
        self._load_edge_classifier()

        # Featurize Image
        sigma_min = 1
        sigma_max = 32
        features_func = partial(
            feature.multiscale_basic_features,
            intensity=False,
            edges=True,
            texture=True,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            channel_axis=None,
        )
        features = features_func(self.image_cropped)

        # Flatten features
        (x, y, z) = features.shape
        features = features.reshape(x * y, z)

        # Predict
        results = future.predict_segmenter(features, self.edge_model)

        # Reshape
        thresh = 255 * (
            results.reshape(x, y).astype(np.uint8) - 1
        )  # To make Overlap and Facet
        return thresh

def edge_classifier(self):
    # Get masking info
    mask = self._edge_pixel_classifier()
    img_logical = mask == 0

    # Get Canny Edge
    canny_args = [(5,5),20,60,80,80,False]
    canny_edge = canny_edge_cv2(self.image,*canny_args)
    img_edges = canny_edge
    self._edge_stats = None

    # Define dead edges
    dead_edges = copy.deepcopy(img_edges)
    dead_edges[img_logical] = 0
    dead_edges[dead_edges > 0] = 255

    # define live edges
    live_edges = copy.deepcopy(img_edges)
    live_edges[~img_logical] = 0
    live_edges[live_edges > 0] = 255

    # broaden edges for visibility, store for figure reference
    dead_broad = cv2.GaussianBlur(dead_edges,(3,3),cv2.BORDER_DEFAULT)
    live_broad  = cv2.GaussianBlur(live_edges,(3,3),cv2.BORDER_DEFAULT)
    color_img = cv2.cvtColor(self.image,cv2.COLOR_GRAY2RGB)
    color_img[dead_broad > 0] = (0,0,255)
    color_img[live_broad > 0] = (0,255,0)
    #color_img[(dead_broad > 0) & (live_broad>0)] = (255,0,0)
    self._edge_highlight = color_img
    self._live_edges = live_edges

    return live_edges
    
def edge_localthresh(self):
    # Get Masking info
    image_blur = cv2.GaussianBlur(self.image,(9,9),cv2.BORDER_DEFAULT)
    thresh = threshold_local(image_blur,35,offset=10)
    mask = self.image > thresh
    mask = quick_close(mask,3,neighbor_threshold=7)
    mask = cv2.dilate(mask.astype(np.uint8),np.ones([3,3]))
    mask = cv2.erode(mask.astype(np.uint8),np.ones([3,3]),iterations=2)

    img_logical = ~mask.astype(bool)
    #img_logical = mask == 0
    #cv2.imwrite(f"localthresh{self._file_name}.png",img_logical.astype(np.uint8)*255)
    
    # Get Canny Edge
    canny_args = [(5,5),20,60,80,80,False]
    canny_edge = canny_edge_cv2(self.image,*canny_args)
    img_edges = canny_edge
    self._edge_stats = None

    # Define dead edges
    dead_edges = copy.deepcopy(img_edges)
    dead_edges[img_logical] = 0
    dead_edges[dead_edges > 0] = 255

    # define live edges
    live_edges = copy.deepcopy(img_edges)
    live_edges[~img_logical] = 0
    live_edges[live_edges > 0] = 255

    # broaden edges for visibility, store for figure reference
    dead_broad = cv2.GaussianBlur(dead_edges,(3,3),cv2.BORDER_DEFAULT)
    live_broad  = cv2.GaussianBlur(live_edges,(3,3),cv2.BORDER_DEFAULT)
    color_img = cv2.cvtColor(self.image,cv2.COLOR_GRAY2RGB)
    color_img[dead_broad > 0] = (0,0,255)
    color_img[live_broad > 0] = (0,255,0)
    #color_img[(dead_broad > 0) & (live_broad>0)] = (255,0,0)
    self._edge_highlight = color_img
    self._live_edges = live_edges

    # Modify threshold
    return live_edges

def edge_testing(self):
    # Get Masking info
    image_blur = cv2.GaussianBlur(self.image,(9,9),cv2.BORDER_DEFAULT)
    thresh = threshold_local(image_blur,35,offset=10)
    mask = self.image > thresh
    mask = quick_close(mask,3,neighbor_threshold=7)
    mask = cv2.dilate(mask.astype(np.uint8),np.ones([3,3]))
    mask = cv2.erode(mask.astype(np.uint8),np.ones([3,3]),iterations=2)

    img_logical = ~mask.astype(bool)
    #img_logical = mask == 0
    #cv2.imwrite(f"localthresh{self._file_name}.png",img_logical.astype(np.uint8)*255)
    
    # Get Canny Edge
    canny_args = [(5,5),10,50,80,80,False]
    canny_edge = canny_edge_cv2(self.image,*canny_args)
    img_edges = canny_edge
    self._edge_stats = None

    # Define dead edges
    dead_edges = copy.deepcopy(img_edges)
    dead_edges[img_logical] = 0
    dead_edges[dead_edges > 0] = 255

    # define live edges
    live_edges = copy.deepcopy(img_edges)
    live_edges[~img_logical] = 0
    live_edges[live_edges > 0] = 255

    # broaden edges for visibility, store for figure reference
    dead_broad = cv2.GaussianBlur(dead_edges,(3,3),cv2.BORDER_DEFAULT)
    live_broad  = cv2.GaussianBlur(live_edges,(3,3),cv2.BORDER_DEFAULT)
    color_img = cv2.cvtColor(self.image,cv2.COLOR_GRAY2RGB)
    color_img[dead_broad > 0] = (0,0,255)
    color_img[live_broad > 0] = (0,255,0)
    #color_img[(dead_broad > 0) & (live_broad>0)] = (255,0,0)
    self._edge_highlight = color_img
    self._live_edges = live_edges
    self._original_thresh = copy.deepcopy(self.thresh)

    # Modify threshold
    return live_edges


def sobel_connectivity(self):
    image_blur = cv2.GaussianBlur(self.image,(9,9),cv2.BORDER_DEFAULT)

    cv2.Sobel()
    
from .dependencies import *
from .settings import *
from .simulation import *
# from slmsuite.holography.analysis import * 

def plot_arrays(arrays, subplot_figsize=(5, 5)):
    num_arrays = len(arrays)
    
    # Create a figure with subplots, one for each array
    fig, axes = plt.subplots(1, num_arrays, figsize=(subplot_figsize[0] * num_arrays, subplot_figsize[1]))
    
    # If there's only one array, axes will not be a list
    if num_arrays == 1:
        axes = [axes]
    
    for i, array in enumerate(arrays):
        im = axes[i].imshow(array, cmap='viridis')
        axes[i].set_title(f'Array {i+1}')
        fig.colorbar(im, ax=axes[i])

    plt.tight_layout()
    plt.show()

def plot_tweezerregions(image, regions, centers, figsize=(10, 10)):
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image, cmap='gray')

    # Get the dimensions of the grid
    grid_height, grid_width = regions.shape[:2]

    for i in range(grid_height):
        for j in range(grid_width):
            # Get the center of the region
            center_x, center_y = centers[i, j]
            
            # Calculate the coordinates of the region's top-left corner
            start_x = int(center_x - regions.shape[2] / 2)
            start_y = int(center_y - regions.shape[3] / 2)
            
            # Create a rectangle patch
            rect = patches.Rectangle((start_x, start_y), regions.shape[2], regions.shape[3],
                                     linewidth=1, edgecolor='r', facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)

            # Plot the center of the region with smaller markers
            ax.plot(center_x, center_y, 'bo', markersize=2)  # Smaller blue dot
    
    plt.show()
    
def get_gridsize(tweezer_array):
    # Find the indices of the points where the array is 1
    points = np.argwhere(tweezer_array == 1)
    
    if points.size == 0:
        raise ValueError("No tweezers found in the array.")
    
    # Determine the number of unique x and y coordinates
    unique_x = len(np.unique(points[:, 1]))
    unique_y = len(np.unique(points[:, 0]))
    
    return unique_x, unique_y

def adjustimage(imagearray, rotateby90=0, fliphorizontal=False, flipvertical=False):
    # Rotate the image by 90 degrees increments
    if rotateby90 not in [0, 1, 2, 3]:
        raise ValueError("rotateby90 must be 0, 1, 2, or 3.")
    if rotateby90 == 1:
        imagearray = np.rot90(imagearray)
    elif rotateby90 == 2:
        imagearray = np.rot90(imagearray, 2)
    elif rotateby90 == 3:
        imagearray = np.rot90(imagearray, 3)
    
    # Flip the image horizontally
    if fliphorizontal:
        imagearray = np.fliplr(imagearray)
    
    # Flip the image vertically
    if flipvertical:
        imagearray = np.flipud(imagearray)
    
    return imagearray

def identifycorners(blobs):
    # Extract coordinates of blobs
    coordinates = np.array([blob.pt for blob in blobs[0]])

    # Calculate the centroid of the points
    centroid = np.mean(coordinates, axis=0)
    
    # Calculate the distance of each point from the centroid
    distances = np.linalg.norm(coordinates - centroid, axis=1)
    
    # Find the indices of the four points with the maximum distance from the centroid
    furthest_indices = np.argsort(distances)[-4:]
    
    # Get the coordinates of the four corner points
    corners = coordinates[furthest_indices]
    
    return corners[np.argsort(corners[:,0])]

def calculateangle(camimage):
    camerablobs = blob_detect(camimage)
    cameracorners = identifycorners(camerablobs)
    angle = math.atan((cameracorners[1,0]-cameracorners[0,0])/(cameracorners[1,1]-cameracorners[0,1]))
    return angle

def rotate_image(image, angle):
    # Calculate the rotation matrix
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])

    # Calculate the translation needed to reposition the rotated image
    center = np.array(image.shape) // 2
    translation = center - np.dot(rotation_matrix, center)

    # Define the affine transformation matrix
    transformation_matrix = np.vstack([np.column_stack([rotation_matrix, translation]),
                                       [0, 0, 1]])

    # Apply the affine transformation to rotate the image
    rotated_image = affine_transform(image, transformation_matrix, mode='constant', cval=0)

    return rotated_image

def definetweezerregions(refinedcamimage, gridsize, N=9):
    # Detect blobs in the refinedcamimage
    blobs = blob_detect(refinedcamimage)
    
    # Identify the corners of the rectangular grid
    corners = identifycorners(blobs)
    
    # Define the grid size
    grid_height, grid_width = gridsize
    
    # Sort corners to get top-left, top-right, bottom-left, bottom-right
    corners = sorted(corners, key=lambda x: (x[1], x[0]))
    top_left, top_right = sorted(corners[:2], key=lambda x: x[0])
    bottom_left, bottom_right = sorted(corners[2:], key=lambda x: x[0])
    
    # Calculate the step size between each grid point
    step_x = (top_right[0] - top_left[0]) / (grid_width - 1)
    step_y = (bottom_left[1] - top_left[1]) / (grid_height - 1)
    
    # Initialize arrays to hold the regions and their centers
    regions = np.empty((grid_height, grid_width, N, N), dtype=refinedcamimage.dtype)
    centers = np.empty((grid_height, grid_width, 2), dtype=np.int32)
    
    for i in range(grid_height):
        for j in range(grid_width):
            # Calculate the ideal center of each region
            ideal_center_x = top_left[0] + j * step_x
            ideal_center_y = top_left[1] + i * step_y
            
            # Round the center coordinates to the nearest integer
            center_x = int(round(ideal_center_x))
            center_y = int(round(ideal_center_y))
            centers[i, j] = [center_x, center_y]
            
            # Define the region's corners
            start_x = max(0, center_x - N // 2)
            start_y = max(0, center_y - N // 2)
            end_x = min(refinedcamimage.shape[1], start_x + N)
            end_y = min(refinedcamimage.shape[0], start_y + N)
            
            # Extract the region from the refinedcamimage
            region = refinedcamimage[start_y:end_y, start_x:end_x]
            
            # If the region is smaller than N x N, pad it with zeros
            if region.shape[0] < N or region.shape[1] < N:
                padded_region = np.zeros((N, N), dtype=refinedcamimage.dtype)
                padded_region[:region.shape[0], :region.shape[1]] = region
                region = padded_region
            
            # Store the region in the regions array
            regions[i, j] = region
    
    return regions, centers

def gaussian_2d(x, y, x0, y0, sigma_x, sigma_y, amplitude, offset):
    return offset + amplitude * np.exp(
        -(((x - x0) ** 2) / (2 * sigma_x ** 2) + ((y - y0) ** 2) / (2 * sigma_y ** 2))
    )

def fit_gaussian_2d(region, method='trf', max_nfev=10000):
    """Fit a 2D Gaussian to the given region."""
    y, x = np.indices(region.shape)
    
    # Find the center coordinates based on maximum pixel value
    y0, x0 = np.unravel_index(np.argmax(region), region.shape)
    
    initial_guess = (
        x0,  # x0 (center x-coordinate)
        y0,  # y0 (center y-coordinate)
        1,   # sigma_x
        1,   # sigma_y
        np.max(region),  # amplitude
        np.min(region)   # offset
    )
    
    try:
        popt, _ = curve_fit(
            lambda xy, x0, y0, sigma_x, sigma_y, amplitude, offset: gaussian_2d(
                xy[0], xy[1], x0, y0, sigma_x, sigma_y, amplitude, offset
            ),
            (x.ravel(), y.ravel()),
            region.ravel(),
            p0=initial_guess,
            method=method,
            max_nfev=max_nfev
        )
        
        fit = gaussian_2d(x, y, *popt)
        residuals = region - fit
        goodness_of_fit = 1 - np.sum(residuals**2) / np.sum((region - np.mean(region))**2)
        return popt, goodness_of_fit
    
    except Exception as e:
        print(f"Error fitting Gaussian: {e}")
        return initial_guess, 0  # Return initial guess and 0 goodness of fit in case of an error

def extract_regions(simulatedcamimage, blob_coordinates, N=18, retrievemax = False):
    """
    Extract N by N regions centered around each blob, fit a Gaussian, and adjust blob coordinates.
    Handle poor fits or abnormally low tweezer power by setting the tweezer power to a fixed value,
    creating a Gaussian tweezer with this power, and setting the goodness of fit to 0.

    Parameters
    ----------
    simulatedcamimage : numpy.ndarray
        The original 2D array containing the image data.
    blob_coordinates : numpy.ndarray
        A 2D numpy array containing the coordinates of blobs.
    N : int
        The size of the region to extract (N by N).

    Returns
    -------
    goodness_of_fit_array : numpy.ndarray
        A 2D numpy array containing the goodness of fit values.
    updated_blob_coordinates : numpy.ndarray
        A 2D numpy array containing the updated blob coordinates.
    regions : numpy.ndarray
        A 2D numpy array where each entry corresponds to an N by N region centered around each updated blob.
    tweezerpower : numpy.ndarray
        A 2D numpy array containing the power of each tweezer.
    """
    num_rows, num_cols = blob_coordinates.shape[:2]
    half_N = N // 2

    regions = np.empty((num_rows, num_cols), dtype=object)
    goodness_of_fit_array = np.zeros((num_rows, num_cols))
    tweezerpower = np.zeros((num_rows, num_cols))
    updated_blob_coordinates = np.zeros_like(blob_coordinates, dtype=float)

    for i in range(num_rows):
        for j in range(num_cols):
            x, y = blob_coordinates[i, j]
            
            # Round coordinates to the nearest integers
            y = np.round(y).astype(int)
            x = np.round(x).astype(int)

            # Ensure the region is within bounds and remains centered
            y_start = max(0, y - half_N)
            y_end = min(simulatedcamimage.shape[0], y + half_N + 1)
            x_start = max(0, x - half_N)
            x_end = min(simulatedcamimage.shape[1], x + half_N + 1)

            # Extract the region and pad if necessary
            region = simulatedcamimage[y_start:y_end, x_start:x_end]
            # if region.shape[0] < N or region.shape[1] < N:
            #     padded_region = np.zeros((N, N))
            #     padded_region[:region.shape[0], :region.shape[1]] = region
            #     region = padded_region


            # Fit a 2D Gaussian to the region
            popt, goodness_of_fit = fit_gaussian_2d(region)
            goodness_of_fit_array[i, j] = goodness_of_fit

            if goodness_of_fit < 0.5:
                print(f"Poor fit at site ({i}, {j})")
                avg_tweezerpower = np.mean(tweezerpower)
                tweezerpower[i, j] = 0.05 * avg_tweezerpower
                goodness_of_fit_array[i, j] = 0
                updated_blob_coordinates[i, j] = blob_coordinates[i, j]
                regions[i, j] = gaussian_2d(*np.indices((N, N)), x0=N//2, y0=N//2, sigma_x=1, sigma_y=1, amplitude=tweezerpower[i, j], offset=0)
            else:
                tweezerpower[i, j] = popt[4]

                # Update the blob coordinates based on the fitted center
                updated_y = y_start + popt[1]
                updated_x = x_start + popt[0]

                updated_blob_coordinates[i, j] = [updated_x,updated_y]
            if retrievemax:
                tweezerpower[i,j] = np.max(region)
                
            y2_start = np.round(max(0, updated_y - half_N)).astype(int)
            y2_end = np.round(min(simulatedcamimage.shape[0], updated_y + half_N + 1)).astype(int)
            x2_start = np.round(max(0, updated_x - half_N)).astype(int)
            x2_end = np.round(min(simulatedcamimage.shape[1], updated_x + half_N + 1)).astype(int)
            
            region = simulatedcamimage[y2_start:y2_end, x2_start:x2_end]
            regions[i,j] = region
            
    return goodness_of_fit_array, updated_blob_coordinates, regions, tweezerpower

def threshold_array(arr, lower_threshold=0.05, upper_threshold=0.95):
    """
    Modify the array such that values greater than the upper threshold are set to 1
    and values less than the lower threshold are set to 0.

    Parameters
    ----------
    arr : numpy.ndarray
        The input array to be modified.
    lower_threshold : float, optional
        The lower threshold, default is 0.05.
    upper_threshold : float, optional
        The upper threshold, default is 0.95.

    Returns
    -------
    numpy.ndarray
        The modified array.
    """
    # Create a copy of the array to avoid modifying the original array
    modified_arr = np.copy(arr)
    
    # Apply the thresholds
    modified_arr[modified_arr > upper_threshold] = 1
    modified_arr[modified_arr < lower_threshold] = 0
    
    return modified_arr

def assign_blob_coordinates(blobs, grid_size):
    # Extract coordinates of detected blobs
    coordinates = np.array([blob.pt for blob in blobs[0]])

    # Sort blobs by y-coordinates first, then by x-coordinates to get initial order
    sorted_indices = np.lexsort((coordinates[:, 0], coordinates[:, 1]))
    sorted_coordinates = coordinates[sorted_indices]

    # Initialize a numpy array to store the coordinates
    blob_coordinates = np.empty(grid_size + (2,), dtype=float)

    # Assign coordinates to the 2D array row by row
    index = 0
    for i in range(grid_size[0]):
        # Sort blobs in each row by x-coordinate
        row_indices = np.argsort(sorted_coordinates[index:index + grid_size[1], 0])
        blob_coordinates[i] = sorted_coordinates[index + row_indices]
        index += grid_size[1]

    return blob_coordinates

# IMPORTANT - Remember how the image conversion for some reason changes the intensity values of the targets. What the fuck?
def assign_tweezer_coordinates(tweezers, grid_size):
    """
    Assigns blob coordinates from an image with blobs as points equal to 1.
    
    Parameters
    ----------
    tweezers : numpy.ndarray
        The 2D numpy array where blobs are represented by points equal to 1.
    grid_size : tuple
        The size of the grid (rows, cols).
        
    Returns
    -------
    blob_coordinates : numpy.ndarray
        A 2D numpy array containing the coordinates of blobs.
    """
    # Find coordinates of points equal to 1
    coordinates = np.argwhere(tweezers > 0.95)
    
    # Sort the coordinates by y (row) and then by x (column)
    sorted_indices = np.lexsort((coordinates[:, 1], coordinates[:, 0]))
    sorted_coordinates = coordinates[sorted_indices]

    # Initialize the blob_coordinates array
    rows, cols = grid_size
    blob_coordinates = np.empty((rows, cols, 2), dtype=int)

    # Assign coordinates to the grid
    for i in range(rows):
        for j in range(cols):
            blob_index = i * cols + j
            if blob_index < len(sorted_coordinates):
                blob_coordinates[i, j] = sorted_coordinates[blob_index]
            else:
                blob_coordinates[i, j] = [-1, -1]  # Fill with invalid coordinates if not enough blobs

    return blob_coordinates

def calculate_distances_blob(blobs, grid_size):
    # Assign blob coordinates to a 2D array
    blob_coordinates = assign_blob_coordinates(blobs, grid_size)
    
    # Initialize arrays to store distances
    horizontal_distances = np.zeros((grid_size[0], grid_size[1] - 1))
    vertical_distances = np.zeros((grid_size[0] - 1, grid_size[1]))

    # Calculate horizontal distances
    for i in range(grid_size[0]):
        for j in range(grid_size[1] - 1):
            horizontal_distances[i, j] = np.linalg.norm(blob_coordinates[i, j + 1] - blob_coordinates[i, j])

    # Calculate vertical distances
    for i in range(grid_size[0] - 1):
        for j in range(grid_size[1]):
            vertical_distances[i, j] = np.linalg.norm(blob_coordinates[i + 1, j] - blob_coordinates[i, j])

    return horizontal_distances, vertical_distances

def calculate_distances_centers(centercoords, grid_size):
    # Assign blob coordinates to a 2D array
    
    # Initialize arrays to store distances
    horizontal_distances = np.zeros((grid_size[0], grid_size[1] - 1))
    vertical_distances = np.zeros((grid_size[0] - 1, grid_size[1]))

    # Calculate horizontal distances
    for i in range(grid_size[0]):
        for j in range(grid_size[1] - 1):
            horizontal_distances[i, j] = np.linalg.norm(centercoords[i, j + 1] - centercoords[i, j])

    # Calculate vertical distances
    for i in range(grid_size[0] - 1):
        for j in range(grid_size[1]):
            vertical_distances[i, j] = np.linalg.norm(centercoords[i + 1, j] - centercoords[i, j])

    return horizontal_distances, vertical_distances

def plot_distance_distribution(horizontal_distances, vertical_distances, bin_size_percentage=5):
    # Flatten distances to compute histogram
    flat_horizontal = horizontal_distances.flatten()
    flat_vertical = vertical_distances.flatten()

    # Calculate bin sizes based on a percentage of the maximum distance in each set
    max_horizontal_distance = np.max(flat_horizontal)
    max_vertical_distance = np.max(flat_vertical)
    min_horizontal_distance = np.min(flat_horizontal)
    min_vertical_distance = np.min(flat_vertical)


    bin_size_horizontal = max_horizontal_distance * (bin_size_percentage / 100.0)
    bin_size_vertical = max_vertical_distance * (bin_size_percentage / 100.0)

    # Calculate bins based on the calculated bin sizes
    bins_horizontal = np.arange(min_horizontal_distance-bin_size_horizontal, max_horizontal_distance + bin_size_horizontal, bin_size_horizontal)
    bins_vertical = np.arange(min_vertical_distance-bin_size_vertical, max_vertical_distance + bin_size_vertical, bin_size_vertical)

    # Plotting horizontal distances histogram
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(flat_horizontal, bins=bins_horizontal, edgecolor='black')
    plt.title('Horizontal Distance Distribution')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.tight_layout(pad=3.0)  # Adjust padding around the plot

    # Plotting vertical distances histogram
    plt.subplot(1, 2, 2)
    plt.hist(flat_vertical, bins=bins_vertical, edgecolor='black')
    plt.title('Vertical Distance Distribution')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.tight_layout(pad=3.0)  # Adjust padding around the plot

    # Display plots
    plt.show()

def maptoslm(inputarray, shape=(1200, 1920)):
    """
    Map inputarray to the center of an output array of specified shape, cropping to shape.

    Parameters
    ----------
    inputarray : numpy.ndarray or cupy.ndarray
        Input 2D array to be mapped.
    shape : tuple, optional
        Desired shape of the output array (default is (1200, 1920)).

    Returns
    -------
    outputarray : numpy.ndarray or cupy.ndarray
        Output array of specified shape with inputarray centered and cropped to shape.
    """
    input_is_cupy = isinstance(inputarray, cp.ndarray)

    # Determine the larger dimension between inputarray and shape
    max_dim = max(inputarray.shape[0], inputarray.shape[1], shape[0], shape[1])

    if input_is_cupy:
        # Convert shape to CuPy array if inputarray is CuPy
        shape = cp.asarray(shape)
        max_dim = cp.asarray(max_dim)
    else:
        max_dim = np.asarray(max_dim)

    # Create an array of size max_dim by max_dim filled with zeros
    if input_is_cupy:
        outputarray = cp.zeros((max_dim, max_dim), dtype=inputarray.dtype)
    else:
        outputarray = np.zeros((max_dim, max_dim), dtype=inputarray.dtype)

    # Calculate the starting indices to place inputarray in the center of outputarray
    start_y = (max_dim - inputarray.shape[0]) // 2
    start_x = (max_dim - inputarray.shape[1]) // 2

    # Place inputarray in the center of outputarray
    if input_is_cupy:
        outputarray[start_y:start_y+inputarray.shape[0], start_x:start_x+inputarray.shape[1]] = inputarray
    else:
        outputarray[start_y:start_y+inputarray.shape[0], start_x:start_x+inputarray.shape[1]] = inputarray

    # Crop outputarray to the specified shape
    outputarray = outputarray[(max_dim -shape[0])//2:(max_dim + shape[0])//2, :]

    return outputarray


def map_cameratotargetint_delta(tweezerpower, uniformtarget, gridsize):
    """
    Map the tweezer power from the camera image onto the target image (uniformtarget)
    at the positions of the dots (value of 1) defined by gridsize.

    Parameters
    ----------
    tweezerpower : numpy.ndarray
        2D numpy array containing the tweezer power values from the camera image.
    uniformtarget : numpy.ndarray
        2D numpy array (same shape as tweezerpower) initially filled with 1s at grid positions.
    gridsize : tuple
        Tuple containing the dimensions (grid_height, grid_width) of the grid.

    Returns
    -------
    mapped_target_image : numpy.ndarray
        The target image (uniformtarget) with updated tweezer power values mapped from the camera image.
    """
    # Get dimensions of the grid
    grid_height, grid_width = gridsize

    # Find the indices of the corners by sorting
    indices = np.argpartition(uniformtarget.ravel(), -grid_height * grid_width)[-grid_height * grid_width:]
    corner_coords = np.unravel_index(indices, uniformtarget.shape)

    # Initialize mapped_target_image as a copy of uniformtarget
    mapped_target_image = np.copy(uniformtarget)

    # Loop through each corner and map tweezer power
    for i in range(grid_height):
        for j in range(grid_width):
            # Determine the corner coordinates
            y_corner, x_corner = corner_coords[0][i * grid_width + j], corner_coords[1][i * grid_width + j]

            # Map tweezer power to the corresponding grid position
            mapped_target_image[y_corner, x_corner] = tweezerpower[i, j]

    return mapped_target_image

def map_regions_to_3x3(regions, numpixels):
    """
    Map each region in the input array 'regions' to a 3 by 3 array by averaging subregions.

    Parameters
    ----------
    regions : numpy.ndarray
        A 2D numpy array where each entry contains an N by N region.
    numpixels : int
        Number of pixels to define the central subregion of size 3 * numpixels.

    Returns
    -------
    shrunkenregions : numpy.ndarray
        A 2D numpy array where each entry is a 3 by 3 array averaged from the subregions of the input regions.
    """

    num_rows, num_cols = regions.shape[:2]
    N = regions[0, 0].shape[0]  # Size of each region

    shrunkenregions = np.empty((num_rows, num_cols), dtype=object)

    for i in range(num_rows):
        for j in range(num_cols):
            region = regions[i, j]

            # Define central subregion
            center = N // 2  # Center of the region
            subregion_size = 3 * numpixels
            subregion_half_size = subregion_size // 2

            # Determine subregion bounds
            y_start = center - subregion_half_size
            y_end = center + subregion_half_size + 1
            x_start = center - subregion_half_size
            x_end = center + subregion_half_size + 1

            # Ensure bounds are within region size
            if y_start < 0:
                y_start = 0
            if y_end > N:
                y_end = N
            if x_start < 0:
                x_start = 0
            if x_end > N:
                x_end = N

            # Extract central subregion
            subregion = region[y_start:y_end, x_start:x_end]

            # Pad subregion if necessary
            if subregion.shape[0] < subregion_size or subregion.shape[1] < subregion_size:
                padded_subregion = np.zeros((subregion_size, subregion_size), dtype=region.dtype)
                padded_subregion[:subregion.shape[0], :subregion.shape[1]] = subregion
                subregion = padded_subregion

            # Calculate 3x3 averaged region
            averaged_region = np.zeros((3, 3), dtype=region.dtype)
            for m in range(3):
                for n in range(3):
                    y_substart = m * numpixels
                    y_subend = y_substart + numpixels + 1
                    x_substart = n * numpixels
                    x_subend = x_substart + numpixels + 1
                    subregion_avg = np.sum(subregion[y_substart:y_subend, x_substart:x_subend])
                    averaged_region[m, n] = subregion_avg

            # Store averaged 3x3 region in shrunkenregions
            shrunkenregions[i, j] = averaged_region

    return shrunkenregions

def map_cameratotargetin_3by3regions(scaledregions, uniformtarget, gridsize):
    """
    Map the tweezer power from the camera image onto the target image (uniformtarget)
    at the positions of the dots (value of 1) defined by gridsize.

    Parameters
    ----------
    tweezerpower : numpy.ndarray
        2D numpy array containing the tweezer power values from the camera image.
    uniformtarget : numpy.ndarray
        2D numpy array (same shape as tweezerpower) initially filled with 1s at grid positions.
    gridsize : tuple
        Tuple containing the dimensions (grid_height, grid_width) of the grid.

    Returns
    -------
    mapped_target_image : numpy.ndarray
        The target image (uniformtarget) with updated tweezer power values mapped from the camera image.
    """
    # Get dimensions of the grid
    grid_height, grid_width = gridsize

    # Find the indices of the corners by sorting
    indices = np.argpartition(uniformtarget.ravel(), -grid_height * grid_width)[-grid_height * grid_width:]
    corner_coords = np.unravel_index(indices, uniformtarget.shape)

    # Initialize mapped_target_image as a copy of uniformtarget
    mapped_target_image = np.copy(uniformtarget)

    # Loop through each corner and map tweezer power
    for i in range(grid_height):
        for j in range(grid_width):
            # Determine the corner coordinates
            y_corner, x_corner = corner_coords[0][i * grid_width + j], corner_coords[1][i * grid_width + j]

            # Map tweezer power to the corresponding grid position
            mapped_target_image[y_corner-1:y_corner+2, x_corner-1:x_corner+2] = scaledregions[i,j]

    return mapped_target_image

# Feedback

def Exp_configureanchors(cam, slm, tweezerGWS, anchorGWS,exposure = 0.02, N = 25, rotateby90=0, fliphorizontal=False, flipvertical=False):
    gridsize = get_gridsize(tweezerGWS.get_uniformtarget())
    anchorphase = anchorGWS.get_slmphase() + np.pi # Because slmphase is currently stored -pi to pi
    anchorphase = maptoslm(anchorphase)
    slm.write(anchorphase, settle=True)
    
    cam.set_exposure(exposure)
    camimage_anchor = adjustimage(cp.array(cam.get_image()), rotateby90, fliphorizontal, flipvertical)
    
    anchorblobs = blob_detect(camimage_anchor)
    corners = identifycorners(anchorblobs)
    angle = calculateangle(camimage_anchor)
    refinedcamimage_anchor = rotate_image(camimage_anchor, angle)
    empty_tweezerregions, empty_tweezercenters = definetweezerregions(refinedcamimage_anchor, gridsize, N)

    
    return angle, refinedcamimage_anchor, empty_tweezerregions, empty_tweezercenters


def Exp_cameratoarray(cam, slm, tweezerGWS, anchorGWS, angle, empty_tweezercenters, exposure, N=25, scalingfactor = 4, retrievemax = False, BloborDelta = 'Delta', rotateby90=0, fliphorizontal=False, flipvertical=False):
    '''mappedblobs - 2D array mapped on uniform targets, tweezerstatistics: 'gaussian fit, tweezer power, hordist, vertdist, 
    cameradata: refined camera image, refined centers'''
    gridsize = get_gridsize(tweezerGWS.get_uniformtarget())
    tweezerphase = tweezerGWS.get_slmphase() + np.pi # Because slmphase is currently stored -pi to pi
    tweezerphase = maptoslm(tweezerphase)
    slm.write(tweezerphase, settle=True)
    
    cam.set_exposure(exposure)
    camimage_tweezer = adjustimage(cp.array(cam.get_image()), rotateby90, fliphorizontal, flipvertical)
    refinedcamimage_tweezer = rotate_image(camimage_tweezer, angle)
    
    goodnessoffit, refined_tweezercenters, tweezerregions, tweezerpower = extract_regions(refinedcamimage_tweezer, empty_tweezercenters, N, retrievemax)
    # More statistics of fit:
    tweezers_hordist, tweezers_vertdist = calculate_distances_centers(refined_tweezercenters, gridsize)
    
    
    # Map camimage to array form
    scaledregions = map_regions_to_3x3(tweezerregions, scalingfactor)
    if BloborDelta == 'Delta':
        mappedblobs = map_cameratotargetint_delta(tweezerpower, tweezerGWS.get_uniformtarget(), gridsize)
    elif BloborDelta =='Blob':
        mappedblobs = map_cameratotargetin_3by3regions(scaledregions, tweezerGWS.get_uniformtarget(), gridsize)
        
    tweezerstatistics =[goodnessoffit, tweezerpower, tweezers_hordist, tweezers_vertdist]
    cameradata = [refinedcamimage_tweezer, refined_tweezercenters]
    
    return mappedblobs, tweezerstatistics, cameradata
    

    
def camerafeedback_tweezers(cam, slm, optimizedout_tweezer, optimizedout_anchors, angle, empty_tweezercenters, exposure_tweezers, costfunction='Exp_DeltaFeedback', iterations = 1, N=25, scalingfactor= 4, retrievemax=False, BloborDelta='Delta', rotateby90 =0, fliphorizontal=False, flipvertical=False, beamtype = "Constant", sigma=1, mu  = 1, magnification = 1):
    tweezerGWS = copy.copy(optimizedout_tweezer)

    uniformtarget = optimizedout_tweezer.get_uniformtarget()
    uniformtarget_out = uniformtarget.copy()
    targetmagnification = cp.shape(uniformtarget)[0] // numpixels
    magnification = targetmagnification * magnification
    slmphase = set_circlemask(expand(optimizedout_tweezer.get_slmphase(), magnification), numpixels * magnification)
    inputbeam = set_circlemask(createbeam(beamtype, numpixels * magnification, sigma, mu), numpixels * magnification)
    outputbeam = set_circlemask(createbeam(beamtype, numpixels, sigma, mu), numpixels)

    if optimizedout_tweezer.get_beam() == None:
        inputbeam = set_circlemask(createbeam(beamtype, numpixels * magnification, sigma, mu), numpixels * magnification)
        outputbeam = set_circlemask(createbeam(beamtype, numpixels, sigma, mu), numpixels)
    else:
        inputbeam = optimizedout_tweezer.get_beam().copy()
        outputbeam = optimizedout_tweezer.get_beam().copy() 
    
    slmplane = join_phase_ampl(slmphase, inputbeam)
    weights = optimizedout_tweezer.get_weights()
    weights_previous = optimizedout_tweezer.get_weightsprev()
    
    
    cutoff = cp.mean(uniformtarget)
    tweezerlocation = cp.where(uniformtarget > cutoff)
    err_maxmindiff = []
    err_uniformity = []
    err_powereff = []
    tweezerstatistics_out = []
    
    startingpower = cp.sum(cp.abs(slmplane)**2)
    fourierplane = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(slmplane), norm="ortho"))
    fourierintensity = cp.square(cp.abs(fourierplane))
    
    ### Make sure we don't change the phase now
    fourierangle = cp.angle(fourierplane)
    
    
    for _ in range(iterations):
        mappedblobs, tweezerstatistics, cameradata = Exp_cameratoarray(cam, slm, tweezerGWS, optimizedout_anchors, angle, empty_tweezercenters, exposure_tweezers, N, scalingfactor, retrievemax, BloborDelta, rotateby90, fliphorizontal, flipvertical)
        stdint = mappedblobs
        weights = costfunction(weights, weights_previous, uniformtarget, stdint)
        weights_previous = weights.copy()
        fourierplane = join_phase_ampl(fourierangle, weights)
        slmplane = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(fourierplane), norm="ortho"))     
        endingpower = cp.sum(cp.abs(slmplane)**2)
        slmplane = cp.multiply(cp.divide(slmplane, endingpower), startingpower)
        slmplane_numpixels = slmplane.copy()
        slmplane_numpixels = cp.mean(slmplane_numpixels.reshape(numpixels, magnification, numpixels, magnification), axis=(-3,-1))
        
        slmphase = undiscretize_phase(discretize_phase(set_circlemask(cp.angle(slmplane_numpixels), numpixels)))
        readout_slmphase = slmphase.copy()
        tweezerGWS.set_slmphase(readout_slmphase)
        
        
        slmplane = join_phase_ampl(expand(slmphase, magnification), inputbeam)     
        
        
        err_maxmindiff.append(Err_MaxMinDiff(stdint, tweezerlocation, uniformtarget))
        err_uniformity.append(Err_Uniformity(stdint, tweezerlocation, uniformtarget))
        err_powereff.append(Err_PowerEff(stdint, tweezerlocation))
        tweezerstatistics_out.append(tweezerstatistics)
        
    errors = [err_maxmindiff, err_uniformity, err_powereff]
    labels = ["MaxMinDiff","Uniformity", "Power Efficiency"]
    inputbeam = cp.mean(inputbeam.reshape(numpixels, magnification, numpixels, magnification), axis=(-3,-1))
    targetintensity = cp.mean(uniformtarget.reshape(numpixels, magnification, numpixels, magnification), axis=(-3,-1))

    
    metrics = tweezerstatistics_out
    tweezerGWS.set_all(readout_slmphase, fourierangle, weights, weights_previous, outputbeam, stdint, uniformtarget_out, uniformtarget, errors, labels)
    cameradata_last = cameradata

    return metrics, cameradata_last, tweezerGWS 
    
    

def Pen_Lukin(w,w_prev,target_im,std_int, harmonicremoval=False, harmoniccoords=[]):
    threshold = cp.mean(target_im)
    if harmonicremoval:
        w[target_im>threshold] = cp.sqrt((cp.mean(std_int[target_im>threshold]) / std_int[target_im>threshold])) * w_prev[target_im>threshold]
        w[harmoniccoords] = 0
    else:
        w[target_im>threshold] =  cp.sqrt(target_im[target_im>threshold] * (cp.mean(std_int[target_im>threshold]) / std_int[target_im>threshold])) * w_prev[target_im>threshold]
    return (w)











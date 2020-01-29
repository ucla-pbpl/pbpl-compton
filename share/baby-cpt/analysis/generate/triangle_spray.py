#https://stackoverflow.com/questions/37117878/generating-a-filled-polygon-inside-a-numpy-array
def check(p1, p2, base_array):
    """
    Uses the line defined by p1 and p2 to check array of 
    input indices against interpolated value

    Returns boolean array, with True inside and False outside of shape
    """
    idxs = np.indices(base_array.shape) # Create 3D array of indices

    p1 = p1.astype(float)
    p2 = p2.astype(float)

    # Calculate max column idx for each row idx based on interpolated line between two points
    if p1[0] == p2[0]:
        max_col_idx = (idxs[0] - p1[0]) * idxs.shape[1]
        sign = np.sign(p2[1] - p1[1])
    else:
        max_col_idx = (idxs[0] - p1[0]) / (p2[0] - p1[0]) * (p2[1] - p1[1]) + p1[1]
        sign = np.sign(p2[0] - p1[0])

    return idxs[1] * sign <= max_col_idx * sign

def create_polygon(shape, vertices):
    """
    Creates np.array with dimensions defined by shape
    Fills polygon defined by vertices with ones, all other values zero"""
    base_array = np.zeros(shape, dtype=float)  # Initialize your array of zeros

    fill = np.ones(base_array.shape) * True  # Initialize boolean array defining shape fill
    # Create check array for each edge segment, combine into fill array
    for k in range(vertices.shape[0]):
        fill = np.all([fill, check(vertices[k-1], vertices[k], base_array)], axis=0)

    # Set all values inside polygon to one
    base_array[fill] = 1

    return base_array
    
def triangle_spray(y_bins, y_lower, y_upper, e_bins, e_lower, e_upper, num):

    # (Row, Col) Vertices of Polygon (Defined Clockwise)
    vertices = []
    for i in range(3):
        vertices.append([random.randint(0, y_bins), random.randint(0, e_bins)])

    vertices = np.array(vertices)

    b = np.array([[1, 1, 1]])
    matrix = np.concatenate((vertices, b.T), axis=1)
    area = np.linalg.det(matrix)
    #print(area)
    if(area>0):
        vertices[[0, 1]] = vertices[[1, 0]] #swap rows
    #print(vertices)
    polygon_array = create_polygon([y_bins, e_bins], vertices)

    # Simple routine to print the final array
    #for row in polygon_array.tolist():
    #    for c in row:
    #        print('{:4.1f}'.format(c), end=",")
    #    print ('')

    M=sparse.coo_matrix(polygon_array)
    M = M.tocsc()
    num_points = M.count_nonzero()

    energies = np.linspace(e_lower, e_upper, e_bins)
    ys = np.linspace(y_lower, y_upper, y_bins)

    ye = np.array([
        [[ys[i], energies[j]] for k in range (int(M[i,j]*num/num_points)+1)] for i, j in zip(*M.nonzero())
        ])
    print(ye.shape)
    return ye.reshape(ye.shape[0]*ye.shape[1], ye.shape[2])
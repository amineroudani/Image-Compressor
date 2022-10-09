#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 23:35:18 2022

@author: amineroudani
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Amine Roudani 

CSE102 PROJECT

"""

import math as math


def ppm_tokenize(stream):
    """Takes an input stream and that returns
    an iterator for all the tokens of stream, 
    ignoring comments.
    
    """
    for line in stream:
        newline = line.split("#")[0]    # We ignore comments
        for token in newline.split():
            yield token
 

def ppm_load(stream):
    """Takes an input stream and that loads the PPM image
    
    Args:
        stream: an input stream.
    Returns:
        A 3-element tuple (w, h, img) where
        w: image width
        h: image height
        img: 2D array containing image pixel information
        
    """
    lst = [x for x in ppm_tokenize(stream)]
    w = int(lst[1])              # We store the image width
    h = int(lst[2])              # We store the image height
    imgData = lst[4:]            # We slice lst to get the pixel info only.
    img = []
    for i in range(h):           # We build the w x h 2D matrix.
        row = []
        for j in range(w):
            row.append((int(imgData[i*w*3 + 3*j]), 
                        int(imgData[i*w*3 + 3*j + 1]), 
                        int(imgData[i*w*3 + 3*j + 2])))
        img.append(row) 
    return (w, h, img)
    

def ppm_save(w, h, img, output):
    """ Takes an output stream and saves the PPM image.
    
    Args:
        w: image width
        h: image height
        img: 2D array containing image pixel informaiton.
        output: an output stream (to which we save the PPM)

    """ 
    with open(output, "w") as out:
        out.write(f'P3 \n {h} {w} \n 255 \n') # We first write the header
        for row in img:                       # Then we write the RBB triplets
            for x in row:
                out.write(f'{x[0]} \t {x[1]} \t {x[2]} \n') 


def clamp(n):
    """ Clamps an integer n to the range [0, ..., 255].

    """
    if n > 255:
        return 255
    elif n < 0:
        return 0
    else:
        return n


def RGB2YCbCr(R, G, B):
    """Takes a point in the RGB color space, 
    converts it to the YCbCr color space, 
    returning the 3-element tuple (Y, Cb, Cr).
    
    """
    Y = 0 + 0.299 * R + 0.587 * G + 0.114 * B
    Cb = 128 - 0.168736 * R - 0.331264 * G + 0.5 * B
    Cr = 128 + 0.5 * R - 0.418688 * G - 0.081312 * B
    return (clamp(round(Y)), clamp(round(Cb)), clamp(round(Cr)))


def YCbCr2RGB(Y, Cb, Cr):
    """Takes a point in the YCbCr color space, 
    converts it to the RGB color space, 
    returning the 3-element tuple (R, G, B).
    
    """
    R = Y + 1.402 * (Cr - 128)
    G = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
    B = Y + 1.772 * (Cb - 128)
    return (clamp(round(R)), clamp(round(G)), clamp(round(B)))


def img_RGB2YCbCr(img):
    """ Takes an image in the RGB-color space and 
    return a 3-element tuple (Y, Cb, Cr) 

    Args:
        img: a 2D array
    Returns:
        A 3-element tuple (Y, Cb, Cr) where Y (resp. Cb, Cr) is a matrix 
        s.t. Y[i][j] (resp. Cb[i][j], Cr[i][j]) denotes the Y (resp. Cb, Cr) 
        component of img[i][j]. 
    
    """
    h = len(img)               # Height of the image
    w = len(img[0])            # Width of the image
    Y  = [[0 for _ in range(w)] for _ in range(h)] 
    Cb = [[0 for _ in range(w)] for _ in range(h)] 
    Cr = [[0 for _ in range(w)] for _ in range(h)]  # Initialize matrices with 0s
    y = 0
    while y < h:                                 # Replace 0s in the matrices
        x = 0                                    # using RGB2YCbCr.
        while x < w:    
            (Y[y][x], Cb[y][x], Cr[y][x]) = RGB2YCbCr(img[y][x][0], 
                                                      img[y][x][1], 
                                                      img[y][x][2])
            x += 1
        y += 1
    return (Y, Cb, Cr)
        

def img_YCbCr2RGB(Y, Cb, Cr):
    """An approximation of the inverse transformation of img_RGB2YCbCr(img)
    
    """
    h = len(Y)
    w = len(Y[0])
    img = [[0 for _ in range(w)] for _ in range(h)] 
    print(img)
    y = 0
    while y < h:
         x = 0
         while x < w:    
             img[y][x] =  YCbCr2RGB(Y[y][x], Cb[y][x], Cr[y][x])   
             x += 1
         y += 1
    return img
 
    
def avg(l):
    """Takes a list of floats or integers and returns the average
    
    """
    return sum(l)/len(l)


def subsampling(w, h, C, a, b):
    """Performs & returns the subsampling of the channel C of size w x h
    in the a:b subsampling mode. 
    
    Args:
        C: 2D array containing image color information.
        w: width of image (int)
        h: height of image (int).
        a: vertical factor for subsampling (int).
        b: horizontal factor for subsampling (int).
    Returns
        S: a 2D-array giving the subsampling of the channel C
    
    """
    remainder_vertical = h % a
    remainder_horizontal = w % b 
    
    # We first fill the remaining area of incomplete blocks with dummy pixels
    # by repeating edge pixels s.t. a (resp. b) divides the width (resp height).
    for row in C:  
        row += [row[-1]] * (w - remainder_horizontal)
    for k in range(h - remainder_vertical):
        C.append(C[-1])
    
    
    width_S = math.ceil(w/b)    # Width and height of the subsampled channel
    height_S = math.ceil(h/a)   # to be returned
    
    # First we are going to compute the averages in the horizontal direction
    tmp = []
    for C_row in C: 
        S_row = []
        for j in range(width_S):
            S_row.append(avg(C_row[b*j:b*(j+1)]))
        tmp.append(S_row)
    
    # Now we have use the tmp matrix which has been averaged horizontally,
    # and we average it vertically in order to obtain our final subsampled 
    # matrix S
    S = [[0 for _ in range(width_S)] for _ in range(height_S)]
    for j in range(width_S):
        c = 0
        for i in range(height_S):
            val = avg([tmp[k][j] for k in range(a * c, a * (c + 1))])
            S[i][j] = int(round(val))
            c += 1
    return S



def extrapolate(w, h, C, a, b):
    """The inverse operation of subsampling: performs & returns the 
    extrapolation of channel C in a:b subsampling mode
    
    Args:
        C: 2D array containing subsampled image color information.
        w: width of image before subsampling (int)
        h: height of image before subsampling (int).
        a: vertical factor for subsampling (int).
        b: horizontal factor for subsampling (int).
    Returns
        A 2D-array giving the extrapolation of the channel C
    
    """
    # First we extrapolate the matrix in the horizontal direction
    tmp = []
    for C_row in C:
        tmp_row = []
        for element in C_row:
            if len(tmp_row) <= w - b:       # We ensure that width of the 
                tmp_row += [element]*b      # extrapolated channel = w
            else:
                tmp_row += [element] * (w%b)
                break
        tmp.append(tmp_row)
    
    # Second we extrapolate the matrix in the vertica; direction
    E = []
    counter = 0
    for row in tmp:
        for k in range(a):
            if counter == h:
                return E
            else:
                E.append(row)
                counter += 1         
    return E



def block_splitting(w, h, C):
    """Takes a channel C and yield all the 8x8 subblocks of the channel, 
    line by line, from left to right, with padding.
    
    Args:
        C: 2D Array
        w: width of C
        h: height of C
        
    Returns:
        A generator yielding all 8x8 subblocks of C, line by line,
        from left to right, with padding. 
    
    """
    
    remainder_vertical = h % 8
    remainder_horizontal = w % 8
    
    # We first fill the remaining area of incomplete blocks
    # with dummy pixels by repeating the edge pixels.
    for row in C:  
        row += [row[-1]] * (8 - remainder_horizontal)
    for k in range(8 - remainder_vertical):
        C.append(C[-1])
      
    # Then we split the channel C into 8x8 subblocks and yield these
    # line by line and from left to right.     
    for y in range(math.ceil(h/8)):
        for x in range(math.ceil(w/8)): 
            yield [row[8*x:8*(x+1)] for row in C[8*y:8*(y+1)]]
  
    
# This function is used below in IDCT        
def C(n):
    """Generates & returns the n x n matrix Cn,
    as defined in the Project guidlines.
    
    Args:
        n: integer giving the length of width of Cn.
    Returns:
        A 2D-array representing the n x n matrix Cn.
    
    """
    M = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            M[i][j] = math.sqrt(2/n) * math.cos((math.pi/n)*(j + 1/2) * i)
    for k in range(n):
        M[0][k] = M[0][k] * math.sqrt(1/2)
    return M


# This function is used several times later in the project
def transpose_matrix(M):
    """Transposes a square matrix M;
    
    Args:
        M: 2D array of equal length and width
        
    Returns:
        Mt: The transpose of M
    
    """
    n = len(M)              
    # M is n x n iff Mt is n x n
    # Initialize Mt with 0s
    Mt = [[0 for _ in range(n)] for _ in range(n)] 
    for i in range(n):
        for j in range(n):
            Mt[j][i] = M[i][j]
    return Mt



# This function is also used several times later in the project
def multiply_matrices(M1, M2):
    """Multiplies two n x n matrices: M1 x M2 and returns the result.
    
    Args:
        M1: 2D array representing a n x n matrix.
        M2: 2D array representing a n x n matrix.
    Returns:
        A 2D array representing the product of M1 and M2.
    
    """
    # M = M1 x M2, initialized with 0s.
    M =  [[0 for _ in range(len(M2[0]))] for _ in range(len(M1))]  
    for i in range(len(M1)):
        for j in range(len(M2[0])):
            for k in range(len(M2)):
                M[i][j] += M1[i][k] * M2[k][j]
    return M



# Below is DCT and IDCT based on the literal expressions. 

def DCT(v):
    """Returns the 1D DCT-II of the vector v.
    
    Args:
        v: 1D array representing a vector v.
    Returns:
        v_hat: 1D array giving the DCT-II of v.
        
    """
    n = len(v)
    dct = []
    for i in range(n):
        v_hat = 0
        for j in range(n):
            v_hat += v[j] * math.cos((math.pi/n)*(j + 1/2)*i)
        v_hat = v_hat * math.sqrt(2/n) 
        dct.append(v_hat)       
    dct[0] = dct[0] * math.sqrt(1/2)
    return dct
  


def IDCT(v):
    """Returns the 1D inverse DCT-II of the vector v.
    
    Args:
        v: 1D array representing a vector v.
    Returns:
        v_hat: 1D array giving the inverse DCT-II of v.
        
    """
    # We multiply 
    n = len(v)
    v_hat = []
    Ct = C(n)
    for i in range(n):
        tmp = 0
        for j in range(n):
            tmp += v[j] * Ct[j][i]
        v_hat.append(tmp)
    return v_hat


def DCT2(m, n, A): 
    """Returns the 2D DCT-II of the m x n matrix A.
    
    Args:
        A: 2D-array representing a matrix A.
        m: height of matrix (integer).
        n: width of matrix (integer).
    Returns:
        a 2D array giving the DCT-II of A.
    
    """
    # Recall that matrix multiplication is associative.
    # We multiply Cm x A x Cn_Transpose
    return multiply_matrices(C(m), multiply_matrices(A, transpose_matrix(C(n))))



def IDCT2(m, n, A):
    """Returns the 2D inverse DCT-II of the m x n matrix A.
   
    Args:
        A: 2D-array representing a matrix A.
        m: height of matrix (integer).
        n: width of matrix (integer).
    Returns:
        a 2D array giving the inverse DCT-II of A.
        
    """
    # We multiply Cm_Transpose x A_hat x Cn
    return multiply_matrices(multiply_matrices(transpose_matrix(C(m)), A), C(n))


def redalpha(i):
    """Takes a non-negative integer i and returns a pair (s, k) s.t.
            s: is an integer in the set {−1,1},
            k: is an integer in the range {0..8},
            and we have cos(i * pi / 16) = s * cos(k * pi / 16)
  
    """
    pi = i // 16
    r = i % 16
    k = min(r, 16 - r)
    s = (-1) ** pi
    if pi < 9:
        return (s, k)
    else:
        return (-s, k)


def ncoeff8(i, j):
    """Takes two integers i & j in range {0..8} and returns a pair (s, k) s.t.
            s: is an integer in the set {−1, 1},
            k: is an integer in the range {0..8}, and
            Cbar[i][j] = cos(k * pi / 16)
    """
    if i == 0:
        return (1, 4)    
    else:
        return (redalpha(i * (2 * j + 1)))


# We precompute the alpha[i]'s / 2.
a = {}
k = 0
while k < 8:
    a[k] = math.cos(k * math.pi / 16) / 2
    k += 1


def AUX_DCT_Chen(v):
    """Auxillary function for DCT_Chen where we compute 
    the columns using the Chen Algorithm. 
    
    Args:
        v: 1D array of length 8
    Returns:
        v_hat: 1D array of length 8.
        
    """
    # We compute things completely manually in order to factorise as much as 
    # possible, thus increasing efficiency.
    v1 = a[4] * (v[0] + v[1] + v[2] + v[3] + v[4] + v[5] + v[6] + v[7])
    v2 = a[1] * (v[0] - v[7]) + a[3] * (v[1] - v[6]) + a[5] * (v[2] - v[5]) + a[7] * (v[3] - v[4])
    v3 = a[2] * (v[0] - v[3] - v[4] + v[7]) + a[6] * (v[1] - v[2] - v[5] + v[6])
    v4 = a[3] * (v[0] - v[7]) - a[7] * (v[1] - v[6]) - a[1] * ( v[2] - v[5]) - a[5] * (v[3] - v[4])
    v5 = a[4] * (v[0] - v[1] - v[2] + v[3] + v[4] - v[5] - v[6] + v[7])
    v6 = a[5] * (v[0] - v[7]) - a[1] * (v[1] - v[6]) + a[7] * (v[2] - v[5]) + a[3] * (v[3] - v[4])
    v7 = a[6] * (v[0] - v[3] - v[4] + v[7]) - a[2] * ( v[1] - v[2] - v[5] + v[6])
    v8 = a[7] * (v[0] - v[7]) + a[5] * (-v[1] + v[6]) + a[3] * ( - v[5] + v[2]) + a[1] * (v[4] - v[3])
    vect = [v1, v2, v3, v4, v5, v6, v7, v8]
    return vect



def DCT_Chen(A):
    """Computes the 2D DCT-II transform of A using the Chen Algorithm.
    
    Args:
        A: 8 x 8 2D array
        
    Returns:
        A_DCT2: 8 x 8 2D array giving the 2D DCT-II transform of A 
    
    """
    M = []
    for row in A:
        M.append(AUX_DCT_Chen(row))
    M_Trans = transpose_matrix(M)
    A_DCT2_Trans = []
    for row in M_Trans:
        A_DCT2_Trans.append(AUX_DCT_Chen(row))
    A_DCT2 = transpose_matrix(A_DCT2_Trans)  
    return A_DCT2



def AUX_IDCT_Chen(v):
    """Auxillary function for IDCT_Chen where we compute 
    the columns using the Chen Algorithm. 
    
    Args:
        v: 1D array of length 8
    Returns:
        v_hat: 1D array of length 8.
        
    """  
    # We compute things completely manually in order to factorise as much as 
    # possible, thus increasing efficiency. 
    v1 = (v[0] + v[4]) * a[4] + v[1] * a[1] + v[2] * a[2] + v[3] * a[3] + v[5] * a[5] + v[6] * a[6] + v[7] * a[7]
    v2 = (v[0] - v[4]) * a[4] + v[1] * a[3] + v[2] * a[6] - v[3] * a[7] - v[5] * a[1] - v[6] * a[2] - v[7] * a[5]
    v3 = (v[0] - v[4]) * a[4] + v[1] * a[5] - v[2] * a[6] - v[3] * a[1] + v[5] * a[7] + v[6] * a[2] + v[7] * a[3]
    v4 = (v[0] + v[4]) * a[4] + v[1] * a[7] - v[2] * a[2] - v[3] * a[5] + v[5] * a[3] - v[6] * a[6] - v[7] * a[1]
    v5 = (v[0] + v[4]) * a[4] - v[1] * a[7] - v[2] * a[2] + v[3] * a[5] - v[5] * a[3] - v[6] * a[6] + v[7] * a[1]
    v6 = (v[0] - v[4]) * a[4] - v[1] * a[5] - v[2] * a[6] + v[3] * a[1] - v[5] * a[7] + v[6] * a[2] - v[7] * a[3]
    v7 = (v[0] - v[4]) * a[4] - v[1] * a[3] + v[2] * a[6] + v[3] * a[7] + v[5] * a[1] - v[6] * a[2] + v[7] * a[5]
    v8 = (v[0] + v[4]) * a[4] - v[1] * a[1] + v[2] * a[2] - v[3] * a[3] - v[5] * a[5] + v[6] * a[6] - v[7] * a[7]
    vect = [v1, v2, v3, v4, v5, v6, v7, v8]
    return vect


def IDCT_Chen(A):
    """Computes the 2D DCT-II transform of A using the Chen Algorithm.
    
    Args:
        A: 8 x 8 2D array
        
    Returns:
        A_IDCT2: 8 x 8 2D array giving the 2D DCT-II inverse transform of A 
    
    """
    M = []
    for row in A:
        M.append(AUX_IDCT_Chen(row))
    M_Trans = transpose_matrix(M)
    A_IDCT2_Trans = []
    for row in M_Trans:
        A_IDCT2_Trans.append(AUX_IDCT_Chen(row))
    A_IDCT2 = transpose_matrix(A_IDCT2_Trans)  
    return A_IDCT2


def quantization(A, Q):
    """Takes two 8x8 matrices of numbers and  
    returns the quantization of A by Q.
    """
    M =  [[0 for _ in range(8)] for _ in range(8)]
    
    for i in range(8):
        for j in range(8):
            M[i][j] = round(A[i][j] / Q[i][j])   
    return M


            
def quantizationI(A, Q):
    """Takes two 8x8 matrices of numbers and  
    returns the inverse quantization of A by Q.
    """
    M =  [[0 for _ in range(8)] for _ in range(8)]
    
    for i in range(8):
        for j in range(8):
            M[i][j] = round(A[i][j] * Q[i][j])   
    return M


LQM = [
  [16, 11, 10, 16,  24,  40,  51,  61],
  [12, 12, 14, 19,  26,  58,  60,  55],
  [14, 13, 16, 24,  40,  57,  69,  56],
  [14, 17, 22, 29,  51,  87,  80,  62],
  [18, 22, 37, 56,  68, 109, 103,  77],
  [24, 35, 55, 64,  81, 104, 113,  92],
  [49, 64, 78, 87, 103, 121, 120, 101],
  [72, 92, 95, 98, 112, 100, 103,  99],
]


CQM = [
  [17, 18, 24, 47, 99, 99, 99, 99],
  [18, 21, 26, 66, 99, 99, 99, 99],
  [24, 26, 56, 99, 99, 99, 99, 99],
  [47, 66, 99, 99, 99, 99, 99, 99],
  [99, 99, 99, 99, 99, 99, 99, 99],
  [99, 99, 99, 99, 99, 99, 99, 99],
  [99, 99, 99, 99, 99, 99, 99, 99],
  [99, 99, 99, 99, 99, 99, 99, 99],
]



def Qmatrix(isY, phi):
    """Takes a boolean isY and a quality factor phi. 
    - If isY is True, returns the standard JPEG quantization matrix 
    for the luminance channel, lifted by the quality factor phi. 
    - If isY is False, returns the standard JPEG quantization matrix 
    for the chrominance channel, lifted by the quality factor phi.
    
    """
    if isY:
        Q = LQM
    else:
        Q = CQM
    if phi >= 50:
        Sphi = 200 - 2*phi 
    else:
        Sphi = round(500/phi)
    for i in range(8):
      for j in range(8):
        Q[i][j] = math.ceil((50 + Sphi * Q[i][j])/100)
    return Q




def zigzag(A):
    """Takes a 8x8 matrix and returns a generator that yields
    all the values of A, following the zig-zag ordering
    
    Args:
        A: 2D array, 8x8
    
    Returns:
        Generator that yields all values of A in zig-zag order
        
    """
    out = [[] for _ in range(15)]
    for i in range(8):      
       for j in range(8):         
            r = i + j      
            if r%2 != 0:
                out[r].append(A[i][j])       
            else:
                out[r].insert(0, A[i][j])
    for row in out:  
        for x in row:     
            yield x
    
    
    
 
def rle0(g):
    """Takes a generator that yields integers and returns a generator that 
    yields the pairs obtained from the RLE0 encoding of g.
       
    Args:
        g: a generator that yields integers 
        
    Returns:
        2-element tuples (i, element) where
        element: is the non-zero integer which the generator yields, and
        i: is an integer that gives the number of 0’s preceding v in the stream.
        
    """
    i = 0
    for element in g:
        if element == 0:
            i += 1
        else:
            yield (i, element)
            i = 0
            

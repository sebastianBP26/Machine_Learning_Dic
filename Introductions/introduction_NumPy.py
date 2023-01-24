import numpy as np

# =============================================================================
# Introduction NumPy Arrays

# Simple array creation
a = np.array([15,26,73,84])

# checking type
type(a)

# Numeric type of elements
a.dtype

# Number of dimensions
a.ndim

# array shape
a.shape

# bytes per element
a.itemsize

# bites of memory used
a.nbytes
# =============================================================================


# =============================================================================

# Multi - dimensional arrays
a = np.array([
    [0,1,2,3],
    [10,11,12,13]
    ])

# Shape (ROW, COLUMNS)
shape = a.shape
print('El array a tiene:', str(shape[0]), 'renglones y', str(shape[1]), 'columnas.')

# Element count
a.size

# number of dimensions
a.ndim

# array shape
a[1,3] # Renglon 1 columna 3

a[1,3] = -1
a

# address second (oneth) row using single index
a[1]
# =============================================================================


# =============================================================================
# Formatting Numerica Display
a = np.arange(1.0, 3.0, 0.5)

a * np.pi

a * np.pi * 1e8

a * np.pi * 1e-6

# User formatting
# set precision
np.set_printoptions(precision = 2)

a * np.pi

a * np.pi * 1e8

a * np.pi * 1e-6

# =============================================================================

# =============================================================================
# Indexing and slicing
# var[lower:upper:step]

a = np.array([10,11,12,13,14])
a[1:3]

# negative indices work also
a[1:-2]

a[-4:3]

# ommiting indices
# omitted boundaries are assumed to be the beginning (or end) of the array

# grab first three elements
a[:3]

# grab last two elements
a[-2:]

# every other element
a[::2]

# array slicing
m = np.arange(0,36).reshape(6,6)

# 1. We want to grab 2,3,4
m[0, 2:5]

# 2. we want 2,8,14,20,26,32
m[:, 2]

# 3. We want 28,29,34,35
m[4:, 4:]

# 4. we want  12, 14, 16 | 24, 26, 28
m[2:5:2, :5:2]
m[2::2, ::2]

m[:, 1:4:2]

# exercise
a = np.arange(25).reshape(5,5)

# array([[ 0,  1,  2,  3,  4],
#        [ 5,  6,  7,  8,  9],
#        [10, 11, 12, 13, 14],
#        [15, 16, 17, 18, 19],
#        [20, 21, 22, 23, 24]])

# 1
# array([[ 0,  x,  2,  x,  4],
#        [ 5,  x,  7,  x,  9],
#        [10,  x, 12,  x, 14],
#        [15,  x, 17,  x, 19],
#        [20,  x, 22,  x, 24]])

a[:, 1:4:2]

# 2
# array([[ 0,  1,  2,  3,  4],
#        [ x,  6,  x,  8,  9],
#        [10, 11, 12, 13, 14],
#        [ x, 16,  x, 18, 19],
#        [20, 21, 22, 23, 24]])

a[1:-1:2, 0:3:2]


# 3
# array([[ 0,  1,  2,  3,  4],
#        [ 5,  6,  7,  8,  9],
#        [10, 11, 12, 13, 14],
#        [15, 16, 17, 18, 19],
#        [x, x, x, x, x]])

a[-1]

# =============================================================================

# sliced arrays shared data

a = np.array([0,1,2,3,4])

# create a slice containing two elements of a
b = a[2:4]

b[0] = 10

# changinb b changed a!
a
np.shares_memory(a,b)

# to fix this you have to use .copy()

# =============================================================================
# Fancy Indexing
# indexing by position
a = np.arange(0,80,10)

# fancy indexing
indices = [1,2,-3]

y = a[indices]

# this also works with setting
a[indices] = 99

# indexing with booleans
# manual cration of masks
mask = np.array([0,1,1,0,0,1,0,0], dtype = bool)

# fancy indexing
y = a[mask]

# Fancy indexing in 2-D
m[[0,1,2,3,4],
  [1,2,3,4,5]]

m[3:, [0,2,5]]

mask = np.array([1,0,1,0,0,1], dtype = bool)
m[mask, 2]

### *** Unlike slicing, fancy indexing creates copies instead of a view
# into original array

# give it a try!
a = np.arange(25).reshape(5,5)

# 1 extract the elements indicated in blue
a[[0,1,2],
  [3,1,0]]

mask = a % 3 == 0
a[mask]

# =============================================================================
# Array Constructor Examples
# Floating point arrays
a = np.array([0,1.0, 2, 3])
a.dtype 
a.nbytes

# Reducing Precision
a = np.array([0,1.,2,3], dtype = 'float32')
a.dtype
a.nbytes

# array shape
a = np.array([0,1,2,3], dtype = 'uint8')
a.dtype
a.nbytes

# =============================================================================
# Array creation functions

# arange
# arange([start,] stop[, step], dtype = None)

np.arange(4)
np.arange(0, 2*np.pi, np.pi/4)
np.arange(1.5, 1.8, 0.3)

# ones, zeros
# ones(shape, dtype = 'float64)
# zeros(shape, dtype = 'float64)

np.ones((2,3), dtype = 'float32')
np.zeros(3)

# array creation functions (cont'd)
# Generate an n by n identity array.
# The default dtype is float64

a = np.identity(4)
a.dtype
np.identity(4, dtype = int)

# empty and full
# empty(shape, dtype = float64, order = 'C')
np.empty(2)

# arraay filled with 5.0
a = np.full(2, 5.0)

# alternative approaches
# slower
a = np.empty(2)
a.fill(4.0)
a

a[:] = 3.0

# Linspace
# generate N evenly spaced elements between (and including) start and
# stop values
np.linspace(0,1, 5)

# logspace
# Generate N evenly spaced elements on a log scale between base**start and
# base** stop (Default base = 10)
np.logspace(0,1,5)

# arrays from/to txt files
# save an array np.savetxt('name.txt', array)

# =============================================================================

# =============================================================================
# Computations with arrays

# Array Calculation Methods
a = np.array([
    [1,2,3],
    [4,5,6]
    ])

# .sum() defaults to adding up all the values in an array.
a.sum()

# supply the keyword axis to sum along the 0th axis.
a.sum(axis = 0)

# supply the keyword axis to sum along the last axis
a.sum(axis = -1)

# other operations on arrays
# Functions work on data passed to it

# Mathematical
# sum, prod, min, max, argmin, argmax, ptp

# Statistics
# mean, std, var

# truth value testing
# any, all

# Min / Max
a = np.array([
    [2,3],
    [0,1]
    ])

# Most NumPy reducers can be used as methods as well as functions
a.min()
np.min(a)

# ARGMIN/MAX
# Many tasks (like optimization) are interested in the location of a min/max
# not the value
a.argmax()

# UNRAVELING
# NumPy includes a function to un-flatten 1D locations
a[np.unravel_index(a.argmax(), a.shape)]
a[np.unravel_index(a.argmin(), a.shape)]

# =============================================================================

# =============================================================================
# Where

# Coordinate locations
# NumPys where functions has two distinct uses. 
# One is to provide coordinates from masks:

a = np.arange(-2,2) ** 2
mask = a % 2 == 0
mask

# Coordinates are returned as a tuple of arrays, one for each axis
np.where(mask)

# Conditional array creation
# Where can also be used to construct a new array by choosing values from 
# other arrays of the same shape.

positives = np.arange(1,5)
negatives = - positives
np.where(mask, positives, negatives)

# Or from scalar values. This can be useful for recoding arrays.
np.where(mask, 1, 0)

# Or from both
np.where(mask, positives, 0)

# Give it a try

a = np.arange(-15,15).reshape(5,6)**2

# 1. The maximum of each row
a.max(axis = 1)

# 2. The mean of each column
a.mean(axis = 0)

# 3. The position of the overall minimum 
np.unravel_index(a.argmin(), a.shape)

# =============================================================================

# =============================================================================
# Array Broadcasting

# NumPy arrays of different dimensionality can be combined in the same expression.
# Arrays with smaller dimension are broadcasted to match the larger arrays,
# without copying data.

# Broadcasting has two rules
# Rule 1: Prepend ones to smaller array's shape
a = np.ones((3,5)) 
b = np.ones(5,)
b.reshape(1,5)
b[np.newaxis, :]

# Rule 2: Dimensions of size 1 are repeated without copying
c = a + b
tmp_b = b.reshape(1,5)

# Broadcasting in action
a = np.array([0,10,20,30])
b = np.array([0,1,2])
y = a[:, np.newaxis] + b

# meshgrid() -> use newaxis appropiately in broadcasting expressions.
# repeat() -> Broadcasting makes repeating an array along a dimension of size 1 unnecessary.

# Meshgrid
x, y = np.meshgrid([1,2], [3,4,5])
z = x + y

# broadcasting: no copies
x = np.array([1,2])
y = np.array([3,4,5])
z = x[np.newaxis, :] + y[:, np.newaxis]

# Broadcasting indices
# Broadcasting can also be used to slice elements from different depths 

data_cube = np.arange(27).reshape(3,3,3)
yi, xi = np.meshgrid(np.arange(3), np.arange(3), sparse = True)
zi = np.array([[0,1,2],
               [1,1,2],
               [2,2,2]])

horizon = data_cube[xi,yi,zi]
# =============================================================================

# Universal Function Methods

#op.reduce(): Applies op to all the elements in a 1-D array a reducing it to a single value. 
a = np.array([1,2,3,4])
np.add.reduce(a) # sum them up

# String list example
a = np.array(['ab', 'cd', 'ef'], dtype = 'object')
np.add.reduce(a)

# Logical op examples
a = np.array([1,1,0,1])
np.logical_and.reduce(a)
np.logical_or.reduce(a)

# summing up each row
a = np.arange(3) + np.arange(0,40,10).reshape(-1,1)
np.add.reduce(a,1)

# sum columns by default
np.add.reduce(a)

# ==================================
# op.accumulate(): It creates a new array containing the intermediate results of the reduce operation
# at each element in a

# add example
a = np.array([1,2,3,4])
np.add.accumulate(a)

# String list example
a = np.array(['ab', 'cd', 'ef'], dtype = 'object')
np.add.accumulate(a)

# Logical op examples
a = np.array([1,1,0,1])
np.logical_and.accumulate(a)
np.logical_or.accumulate(a)

# ==================================
# op.reduceat(): It applies op to a ranges in the 1-D array a defined by the values in indices.
# The resulting array has the same length as indices

a = np.array([0,10,20,30,40,50])
indices = np.array([1,3])
np.add.reduceat(a, indices)

# ==================================
# op.outer(): It forms all possible combinations of elements between a and b using op.
# The shape of the resulting array results from concatenating the sahpes of a and b. 

a = np.array([1,2,3,4])
b = np.array([1,2,3])

np.add.outer(a,b)
np.add.outer(b,a)

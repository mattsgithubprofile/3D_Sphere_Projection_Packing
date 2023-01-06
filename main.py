import cv2 as cv
import numpy as np
import random

from matplotlib import pyplot as plt

# this script uses numerical methods to distribute spheres in 3D space in such a manner
# that three orthographic projections result in images of relative closely packed circles

# the method used here does not guarantee any particular density of spheres in the final result
# or distribution of different sizes of spheres

# this is just a pretty naiive algorithm and the implementation isn't even optimized at all
# so don't get too excited

# define the cube size
CUBE_SIZE = 256

# define the minimum and maximum sphere radii
R_MIN = 5
R_MAX = 100

# for each view, keep track of which pixels are filled by any spheres
# {0: empty, 255: filled}
# also keep track of pixels where we know we cannot place any spheres
# {1: sphere can be placed there (maybe), 0: sphere cannot be placed there}
front_view_filled = np.zeros((CUBE_SIZE,CUBE_SIZE), dtype=np.uint8)
front_view_candidate = front_view_filled.copy().astype(np.uint8)
front_view_candidate[R_MIN:-R_MIN,R_MIN:-R_MIN] = 1
right_view_filled = front_view_filled.copy()
right_view_candidate = front_view_candidate.copy()
top_view_filled = front_view_filled.copy()
top_view_candidate = front_view_candidate.copy()

# each sphere will be stored as [xpos, ypos, zpos, radius]
spheres = []

# cv.imshow('front',front_view_candidate)
# cv.waitKey(0)

# keep looping through placing spheres until there are no more spheres that can be placed
while np.max(front_view_candidate):

    # randomly pick any candidate pixel in the front view to attempt to place a sphere

    # find how many pixels are available in each column of front view for possible sphere placement
    col_counts = np.sum(front_view_candidate,axis=0)
    # decide which pixel to attempt to place a sphere
    candidate_pixel_ind = random.randrange(0,np.sum(col_counts))
    # find which column this pixel is in
    col_counts_cumsum = np.cumsum(col_counts)
    front_col_ind = np.searchsorted(col_counts_cumsum,candidate_pixel_ind,side='right')
    # find which row this pixel is in
    candidate_pixel_ind -= col_counts_cumsum[front_col_ind-1]
    front_row_ind = np.searchsorted(np.cumsum(front_view_candidate[:,front_col_ind]),candidate_pixel_ind,side='right')

    # determine whether there are spots in the appropriate row of both the right and top images for this sphere to fit

    right_candidate_indices = np.argwhere(right_view_candidate[front_row_ind,:])
    top_candidate_indices = np.argwhere(top_view_candidate[front_col_ind,:])
    candidate_indices = np.intersect1d(right_candidate_indices,top_candidate_indices)
    if candidate_indices.size:
        # if there is a place where a sphere can fit, randomly pick a spot
        right_col_ind = candidate_indices[random.randrange(0,candidate_indices.size)]

        # determine the largest sphere that can fit in this spot (naiively)

        # max radius to even test based on parameter and fitting in cube
        this_max_radius = min(R_MAX, front_col_ind, CUBE_SIZE-1-front_col_ind, front_row_ind, CUBE_SIZE-1-front_row_ind, right_col_ind, CUBE_SIZE-1-right_col_ind)
        # test if sphere fits with given radius and iteratively reduce it in size until it fits
        while this_max_radius > R_MIN:
            this_sphere_front = np.zeros((CUBE_SIZE,CUBE_SIZE), dtype=np.uint8)
            this_sphere_right = this_sphere_front.copy()
            this_sphere_top = this_sphere_front.copy()
            # draw the sphere as it will appear in the front image at this size
            this_sphere_front = cv.circle(this_sphere_front,(front_col_ind,front_row_ind),this_max_radius,255,-1)
            # check for intersection of this sphere with other spheres already in front view
            if np.max(cv.bitwise_and(this_sphere_front,front_view_filled)):
                # reduce the radius by 1 and try again
                this_max_radius -= 1
                continue
            # repeat check for right and top views
            this_sphere_right = cv.circle(this_sphere_right,(right_col_ind,front_row_ind),this_max_radius,255,-1)
            if np.max(cv.bitwise_and(this_sphere_right,right_view_filled)):
                this_max_radius -= 1
                continue
            this_sphere_top = cv.circle(this_sphere_top,(right_col_ind,front_col_ind),this_max_radius,255,-1)
            if np.max(cv.bitwise_and(this_sphere_top,top_view_filled)):
                this_max_radius -= 1
                continue
            # it didn't fail any of the checks, so break out of sphere shrinking loop
            break
            
        # save this sphere
        spheres.append([front_col_ind, front_row_ind, right_col_ind, this_max_radius])

        # add this sphere to the images
        cv.circle(front_view_filled,(front_col_ind,front_row_ind),this_max_radius,255,-1)
        cv.circle(right_view_filled,(right_col_ind,front_row_ind),this_max_radius,255,-1)
        cv.circle(top_view_filled,(right_col_ind,front_col_ind),this_max_radius,255,-1)

        # mark out any spots where we know a new sphere cannot be placed now
        # draw the circle with R_MIN added radius
        cv.circle(front_view_candidate,(front_col_ind,front_row_ind),this_max_radius+R_MIN,0,-1)
        cv.circle(right_view_candidate,(right_col_ind,front_row_ind),this_max_radius+R_MIN,0,-1)
        cv.circle(top_view_candidate,(right_col_ind,front_col_ind),this_max_radius+R_MIN,0,-1)

        # cv.imshow('a',np.vstack([np.hstack([cv.rotate(top_view_filled, cv.ROTATE_90_COUNTERCLOCKWISE),top_view_filled*0]),np.hstack([front_view_filled,right_view_filled])]))
        cv.imshow('b',front_view_candidate*255)
        cv.waitKey(1)
    else:
        # the spot doesn't fit anywhere, so mark the front view pixel as dead    
        front_view_candidate[front_row_ind,front_col_ind] = 0

# plot the spheres

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
for sphere in spheres:
    ax.plot_wireframe(x*sphere[3]+sphere[0], y*sphere[3]+sphere[1], z*sphere[3]+sphere[2], color="r")

ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
ax.set_proj_type('ortho')

plt.show()

pass

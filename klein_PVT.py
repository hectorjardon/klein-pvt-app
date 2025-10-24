# python -m streamlit run klein_PVT.py is the command to run the file in the powershell


# main.py
import streamlit as st
import numpy as np
import numpy.linalg as la   
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.spatial import ConvexHull

st.set_page_config(page_title="Hyperbolic PVT sampler", layout="centered")

st.title("Hyperbolic PVT sampler")
st.markdown("When you click **Generate** a sample from the Poisson-Voronoi tessellation on hyperbolic plane will be plotted on the Klein model.")

# Parameter inputs (change ranges/defaults to match your needs)
a = st.number_input("Intensity", value=1.0, step=0.01, format="%.3f")
b = st.number_input("Hyperbolic radius", value=3.0, step=1.0, format="%.1f")
c = st.number_input("Tolerance", value=0.9, step=0.01, format="%.2f")

#Code generating the function and plot


# We want to compute the hyperbolic distance between two points of the unit disk in the Poincaré disk model.
# Designed for input of arrays (n,2)
def hyperbolic_distance(p1,p2):
    assert (la.norm(p1,axis=1) < 1).all() and (la.norm(p2,axis=1) < 1).all(), "Points must be inside the unit disk"
    euc_dist = la.norm(p1 - p2,axis=1)**2
    denom = (1 - la.norm(p1,axis=1) **2) * (1 - la.norm(p2,axis=1)**2)
    return np.arccosh(1 + 2 * euc_dist / denom)
# Maybe useful to use the arctanh with complex numbers?

# Generate Poisson point process in the Poincaré disk
def poisson_points_hyp(intensity,radius): # intensity, radius of the disk of sampling in hyperbolic metric
    num_points = np.random.poisson(intensity * 2*np.pi * (np.cosh(radius)-1)) # number of points in the disk, Poison (intensity*area)
    radii_coordinates = np.random.uniform(low=0, high=1, size=num_points) # sample num_points from uniform in [0,1]
    radii = np.tanh(np.arccosh(1 + radii_coordinates*(np.cosh(radius)-1))/2) # sample radii by inverse transform sampling
    angles = np.random.uniform(low=0, high=2*np.pi, size=num_points) #angles sampled form uniform in [0,2pi]
    return np.array([radii*np.cos(angles),radii*np.sin(angles)]).T 
#output is points in Poincare disk with shape (num_points,2), for compatibility with Delaunay


def circumcenters_2d(triangles):

    A = triangles[:, 0, :] #Each A, B, C correspond to the vertices of the triangles
    B = triangles[:, 1, :]
    C = triangles[:, 2, :]
    
    # Compute determinants
    D = 2 * (A[:,0]*(B[:,1]-C[:,1]) + B[:,0]*(C[:,1]-A[:,1]) + C[:,0]*(A[:,1]-B[:,1]))
    
    # Circumcenter coordinates
    U = ((A[:,0]**2 + A[:,1]**2)*(B[:,1]-C[:,1]) +
         (B[:,0]**2 + B[:,1]**2)*(C[:,1]-A[:,1]) +
         (C[:,0]**2 + C[:,1]**2)*(A[:,1]-B[:,1])) / D

    V = ((A[:,0]**2 + A[:,1]**2)*(C[:,0]-B[:,0]) +
         (B[:,0]**2 + B[:,1]**2)*(A[:,0]-C[:,0]) +
         (C[:,0]**2 + C[:,1]**2)*(B[:,0]-A[:,0])) / D
    
    circumcenters = np.stack([U, V], axis=1)
    
    # Circumradius: distance from circumcenter to any vertex (here A)
    circumradii = np.linalg.norm(circumcenters - A, axis=1)
    
    return circumcenters, circumradii

# The following piece of code is used to clean 

# The following gets a Euclidean circle in the unit disk, and returns the parameters of the corresponding hyperbolic circle in the Poincare disk model
def euc_to_hyp(c, R):
    #First get the norms
    norms = la.norm(c, axis=1)
    
    # First we normalize the directions,
    c_unit = c / norms[:, np.newaxis]

    # Then compute the hyperbolic radius of the circles
    r = hyperbolic_distance(c+R[:,np.newaxis]*c_unit , c- R[:,np.newaxis]*c_unit) / 2

    # Auxiliar quantity
    t = np.tanh(r / 2)

    # Hyperbolic center
    hyp_norms = np.sqrt((t**2 + norms**2 - R**2) / (1 + t**2 *(norms**2 - R**2)))
    a = hyp_norms[:, np.newaxis] * c_unit
    
    return a

def poincare2klein(p):
    #Input is (n,2)
    norm_p2 = np.sum(p**2, axis=-1, keepdims=True)
    k = (2 * p) / (1 + norm_p2)
    return k

# You give it the points and the triangles, it returns a list of polygons corresponding to the Voronoi cells
def compute_patches(points, circumcenters, triangles):
    mask = (triangles[:,:,np.newaxis] == range(len(points))).any(axis=1) # mask to identify triangles containing each point
    hulls = [ConvexHull(circumcenters[mask[:,i],:]) for i in range(len(points)) if np.sum(mask[:,i])>=3] # only keep cells with at least 3 vertices
    patches = [Polygon(hull.points[hull.vertices],closed = True) for hull in hulls] # list of polygons for the cells
    return patches


def plot_Klein_PVT(intensity,radius,tol):
    points = poisson_points_hyp(intensity,radius + np.arccosh(1 - np.log(1 - tol)/(2*np.pi*intensity))) # add margin to radius to avoid edge effects, ensures that with probability tol a point will be sampled in this windo
    if len(points) > 30000:   # choose your threshold
        raise ValueError(f"Oops! those are {len(points)} points in the Poisson sample and we do not want anything to crash. Try reducing some parameter.")
    tri = Delaunay(points) # Delaunay triangulation of the points
    triangles_idx = tri.simplices # indices of the triangles
    triangles_physical = points[triangles_idx] # shape (num_triangles,vertices,dimensions) where m is the number of triangles
    circumcenters, circumradii = circumcenters_2d(triangles_physical) # shape (num_triangles,2), (num_triangles,)
    inside_idx = (circumradii + la.norm(circumcenters,axis=1))<1 # only keep circumcenters for disks inside the unit disk
    hyp_circumcenters = euc_to_hyp(circumcenters[inside_idx,:],circumradii[inside_idx]) #hyperbolic circumcenters of the triangles with circumcenters inside the unit disk
    
    triangles_idx_good = triangles_idx[inside_idx,:] # only keep triangles with circumcenters inside the unit disk
    mask = (triangles_idx_good[:,:,np.newaxis] == range(len(points))).any(axis=1) # mask to identify triangles containing each point
    kvxs = poincare2klein(hyp_circumcenters) # circumcenters in the Klein model
    hulls = [ConvexHull(kvxs[mask[:,i],:]) for i in range(len(points)) if np.sum(mask[:,i])>=3] # only keep cells with at least 3 vertices
    patches = [Polygon(hull.points[hull.vertices],closed = True) for hull in hulls] # list of polygons for the cells
    colors = np.random.rand(len(points),4) # random colour for each cell
    collection = PatchCollection(patches, facecolor=colors, edgecolor='black', alpha=0.7)

    # Plot
    poin_radius = np.tanh(radius/2) # radius of the Poincare disk 
    w = 2 * poin_radius / (1 + poin_radius**2) # radius of the Klein disk
    fig, ax = plt.subplots()
    ax.add_collection(collection)
    ax.set_xlim(-w, w)
    ax.set_ylim(-w, w)
    ax.set_aspect('equal','box')
    ax.axis('off')
    ax.set_title(f"Intensity = {intensity}, Radius = {radius}, Tolerance = {tol}")
    return fig, ax

# Button to generate
if c >= 1 or c <= 0:
    st.error("Error: tolerance must be strictly less than 1.")
else:
    if st.button("Generate plot"):
        try:
            fig, ax = plot_Klein_PVT(a, b, c)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Error generating plot: {e}")

    else:
        st.info("Change parameters and click **Generate plot**.")



























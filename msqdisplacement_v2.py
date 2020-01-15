#!/usr/bin/env python
import numpy as np
import math
import sys
supercell_size = [4,1,2]
filename = 'POSCAR_0GPa_BH2'
src = filename + '_0pc_src_4'
dest = filename + '_0pc_src_32'


with open(src) as sourcefile:
    lines = sourcefile.readlines()
    src_latt = (lines[2].split(), lines[3].split(), lines[4].split())
    src_latt = np.array(src_latt,dtype=np.float32)
    src_atoms = np.array(lines[6].split(),dtype=np.int)
    print("Source cell:", src_atoms)
    n_src_atoms = sum(src_atoms)
    src_coord = []
    for i, line in enumerate(lines):
        if i>=8 and i<(8+n_src_atoms):
            src_coord.append(line.split())
    ### GROUPS OF ATOMS ###
    src_group_1 = np.array(src_coord[0:src_atoms[0]],dtype=np.float32)
    slider = src_atoms[0] 
    src_group_2 = np.array(src_coord[slider:slider+src_atoms[1]],dtype=np.float32) 
    slider = slider + src_atoms[1] 
    src_group_3 = np.array(src_coord[slider:],dtype=np.float32)
##    print(src_group_1)
##    print(src_group_2)
##    print(src_group_3)
    src_coord = np.array(src_coord,dtype=np.float32)
with open(dest) as destfile:
    lines = destfile.readlines()
    dest_latt = (lines[2].split(), lines[3].split(), lines[4].split())
    dest_latt = np.array(dest_latt,dtype=np.float32)
    dest_atoms = np.array(lines[6].split(),dtype=np.int)
    print("Dest cell:", dest_atoms)
    n_dest_atoms = sum(dest_atoms)
    dest_coord = []
    for i, line in enumerate(lines):
        if i>=8 and i<(8+n_dest_atoms):
            dest_coord.append(line.split())
    ### GROUPS OF ATOMS ###
    dest_group_1 = np.array(dest_coord[0:dest_atoms[0]],dtype=np.float32)
    slider = dest_atoms[0] 
    dest_group_2 = np.array(dest_coord[slider:slider+dest_atoms[1]],dtype=np.float32) 
    slider = slider + dest_atoms[1] 
    dest_group_3 = np.array(dest_coord[slider:],dtype=np.float32)
##    print(dest_group_1)
##    print(dest_group_2)
##    print(dest_group_3)
    dest_coord = np.array(dest_coord,dtype=np.float32)

### FUNCTION, CALCULATES EUCLIDEAN DISTANCE
def euclide(atom1, atom2):
    distance = np.linalg.norm(atom1-atom2)
    return distance

supercell_param = np.array(supercell_size)
## THE ORIGINAL CELL WILL BE TRANSLATED BY AN AMOUNT EQUAL TO A FRACTION OF THE NEW AXES ##
unit_latt = (dest_latt/supercell_param)
## DIFFERENCE OF THE LATTICES (fraction of doped cell and original cell)
##print(unit_latt - src_latt)


### ITERATION ON SUPERCELLS ###
delta_u = 0.000
for iter_x in range(supercell_size[0]):
    for iter_y in range(supercell_size[1]):
        for iter_z in range(supercell_size[2]):
            print("Acting on cell:", iter_x+1, iter_y+1, iter_z+1)
            ### TRANSLATION OF THE ORIGIN OF THE ATOMS
            ### EQUAL TO: cell sublattice times supercell factors 
##            print(unit_latt*[iter_x, iter_y, iter_z])
            origin = sum(np.array(unit_latt.T * np.array([iter_x, iter_y, iter_z]), dtype=np.float32).T)
##            print(origin)
            ### DEFINE GROUPS OF COORDINATES ###
            ### GROUPS for translated cell
            group_1 = src_group_1 @ unit_latt + origin
            group_2 = src_group_2 @ unit_latt + origin
            group_3 = src_group_3 @ unit_latt + origin
            ### D_GROUPS for doped cell
            d_group_1 = dest_group_1 @ dest_latt
            d_group_2 = dest_group_2 @ dest_latt
            d_group_3 = dest_group_3 @ dest_latt
            ### NOW FOR EACH GROUP FIND THE MATCHING COORDINATES USING EUCLIDEAN DISTANCE
            for atom1 in group_1:
                min_dist = math.inf
                for atom2 in d_group_1:
                    dist = euclide(atom1, atom2)
                    if dist < min_dist:
                        min_dist = dist
                delta_u += min_dist
#                print(min_dist)
            for atom1 in group_2:
                min_dist = math.inf
                for atom2 in d_group_2:
                    dist = euclide(atom1, atom2)
                    if dist < min_dist:
                        min_dist = dist
                delta_u += min_dist
#                print(min_dist)
            for atom1 in group_3:
                min_dist = math.inf
                for atom2 in d_group_3:
                    dist = euclide(atom1, atom2)
                    if dist < min_dist:
                        min_dist = dist
                delta_u += min_dist
#                print(min_dist)
delta_u = delta_u/(3*n_dest_atoms)
print("\nMean squared displacement, in A^-2   ", delta_u*100)

###print(dest_atoms.shape)
###print(dest_latt.shape)
###print(dest_coord.shape)

n_mult = supercell_size[0]*supercell_size[1]*supercell_size[2]
if n_dest_atoms != n_src_atoms*n_mult:
    sys.exit("N. of atoms in the source is not equal to n. of atoms in the dest! Abort!\n")

#thr = 0.8
#diff_coord_latt = src_coord - dest_coord
#mask_min = diff_coord_latt > thr
#mask_plus = diff_coord_latt < -thr
#src_coord[mask_min] = src_coord[mask_min]-1
#src_coord[mask_plus] = src_coord[mask_plus]+1
####diff_coord_latt = src_coord - dest_coord
####mask = mask_min + mask_plus
####print(diff_coord_latt[mask])
#
#n_atoms = n_dest_atoms
#src_coord_cart = np.dot(src_coord,src_latt)
#dest_coord_cart = np.dot(dest_coord,dest_latt)
###print(src_coord_cart)
###print(dest_coord_cart)
#
#diff_coord_cart = src_coord_cart - dest_coord_cart
#diff_coord_cart_sq = np.square(diff_coord_cart)
####print("Component-wise sum of squared values:", sum(diff_coord_cart_sq)/n_atoms, "Angstrom squared")
####print("Sum of all squared values:", sum(sum(diff_coord_cart_sq)/n_atoms), "Angstrom squared")
#msqdisp = np.sqrt(sum(sum(diff_coord_cart_sq)/n_atoms))
#print("Mean squared displacement:", msqdisp, "A")

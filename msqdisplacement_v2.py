#!/usr/bin/env python

### TUNABLE PARAMETERS ###
supercell_size = [4,1,4]
filename = 'POSCAR_0GPa_BH2'
src = filename + '_0pc_src_4'
dest = filename + '_0pc_src_64.vasp'
####################################################
############  HOW TO USE THIS SCRIPT ###############
####################################################
### This script is still in a rather bad shape, sorry if the code is not 100% clear.
### I am the only one working on it, and it's not the main focus of my work.
### In short, what this script does is: reads a "source" POSCAR file, which is the 
### undistorted cell whence the supercell was generated. It receives in input the 
### "supercell size", in terms of the source cell (i.e. how many repetitions of the
### source cell were used). Then it reads the POSCAR file of the relaxed supercell.
### The algorithm reads the lattice vectors of the supercell and divides them by the 
### supercell size. Then it takes the starting unit cell, shifts it by the proper fraction
### of the supercell axes, and compares the distance between the starting and the final position.
### Since the order in which the coordinates are given may have changed, 
### it compares atoms of the same type, and considers pairs whose distance is minimum.
### The distance is calculated as Euclidean distance, summed over all pairs and divided by 3N
### where N is the number of atoms in the supercell.

### THIS CODE WAS WRITTEN FOR Ca B H, SO IT IS IMPLEMENTED ONLY FOR A CELL WITH THREE ATOMIC SPECIES!
### When doping it is enough to modify your POSCAR file so that all the substituted atoms are listed 
### as if they were the original atoms. E.g.
### Na  Ca  B  H
###  1  31  64 128
### Is transformed into
### Ca  B  H
### 32  64 128
### So that "Na" is considered as "Ca" for the sake of calculting distances.

### I would love to have the time to generalize it, but I don't. I have a PhD to finish!
### IF YOU WANT TO USE IT FOR YOURSELF FIRST TEST IT. Make a supercell without relaxing it. The code 
### should give approximately zero if you do.

import numpy as np
import math
import sys


### Reads source file, lattice vectors, atoms contained and number of atoms
### Do it twice: for "source" POSCAR and for "dest" POSCAR (dest is the "supercell" POSCAR)
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
    ### Grouping atoms greatly speeds up the calculation (less comparisons) and guarantees robustness
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

### ROUTINE CHECK:
### if the number of atoms in the starting and final cell does not match, there is something wrong.
n_mult = supercell_size[0]*supercell_size[1]*supercell_size[2]
if n_dest_atoms != n_src_atoms*n_mult:
    sys.exit("N. of atoms in the source is not equal to n. of atoms in the dest! Abort!\n")

### FUNCTION, CALCULATES EUCLIDEAN DISTANCE
def euclide(atom1, atom2):
    distance = np.linalg.norm(atom1-atom2)
    return distance

supercell_param = np.array(supercell_size)
## THE ORIGINAL CELL WILL BE TRANSLATED BY AN AMOUNT EQUAL TO A FRACTION OF THE NEW AXES ##
## 
unit_latt = (dest_latt.T/supercell_param).T
## DIFFERENCE OF THE LATTICES (fraction of doped cell and original cell)
##print(unit_latt - src_latt)
##w1=1.0
##w2=1-w1
##unit_latt = (w1*unit_latt + w2*src_latt)/(w1+w2)

### ITERATION ON SUPERCELLS ###
delta_u = 0.000
for iter_x in range(supercell_size[0]):
    for iter_y in range(supercell_size[1]):
        for iter_z in range(supercell_size[2]):
            print("Acting on cell:", iter_x+1, iter_y+1, iter_z+1)
            ### TRANSLATION OF THE ORIGIN OF THE ATOMS
            ### EQUAL TO: cell sublattice times supercell factors 
##            print(unit_latt*[iter_x, iter_y, iter_z])
#            print(unit_latt)
##            print(unit_latt.T)
            origin_matrix = np.array(unit_latt.T * np.array([iter_x, iter_y, iter_z]), dtype=np.float32).T
#            print(origin_matrix)
            origin = sum(origin_matrix)
##            print(origin)
            ### DEFINE GROUPS OF COORDINATES ###
            ### GROUPS for translated cell
            ### This trick with masks is used to check for mismatches due to periodic
            ### boundary conditions
            ### e.g. if in a simple cubic an atom is in position (5.1,5.0,5.0) when is expected to be in (0.1,0.0,0.0))
            group_1 = src_group_1 @ unit_latt + origin 
            group_2 = src_group_2 @ unit_latt + origin 
            group_3 = src_group_3 @ unit_latt + origin
            mask_a = ( group_1 >= (sum(unit_latt + origin_matrix)))
            mask_b = ( group_1 <= (origin))
            if (mask_a.size != 0 or mask_b.size != 0):
                print("Mismatch in periodic boundary conditions. Check coordinates:", group_1[mask_a], group_1[mask_b])           
            mask_a = ( group_2 >= (sum(unit_latt + origin_matrix) ))
            mask_b = ( group_2 <= (origin))
            if (mask_a.size != 0 or mask_b.size != 0):
                print("Mismatch in periodic boundary conditions. Check coordinates:", group_2[mask_a], group_2[mask_b])           
            mask_a = ( group_3 >= (sum(unit_latt + origin_matrix) ))
            mask_b = ( group_3 <= (origin))
            if (mask_a.size != 0 or mask_b.size != 0):
                print("Mismatch in periodic boundary conditions. Check coordinates:", group_3[mask_a], group_3[mask_b])           

            ### D_GROUPS for doped cell
            d_group_1 = dest_group_1 @ dest_latt
            d_group_2 = dest_group_2 @ dest_latt
            d_group_3 = dest_group_3 @ dest_latt
##            print(group_1)
##            print(d_group_1)
            ### NOW FOR EACH GROUP FIND THE MATCHING COORDINATES USING EUCLIDEAN DISTANCE
            for atom1 in group_1:
                min_dist = math.inf
                for atom2 in d_group_1:
                    dist = euclide(atom1, atom2)
                    if dist < min_dist:
                        min_dist = dist
                delta_u += min_dist
 #               print(min_dist)
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

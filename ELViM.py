#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 07:50:54 2023

@author: rafael
"""

import argparse
import numpy as np
from numba import njit, prange
import os
import shutil
import time
import mdtraj as md
import force_scheme as force
import sys

@njit(parallel=True, fastmath=False, cache=True)         ####dissimilarity defautl, generalizar para diferentes sigma_0 e epsilon
def square_to_condensed(sqdmat):
   """
   Convert the symmetrical square matrix to the required condensed form.
   """
   size, _ = sqdmat.shape
   total = size*(size+1)//2
   d=np.zeros(total, dtype=np.float32) 
   for k in prange (size):
      for l in range (k, size):
         d[int(total - ((size - k) * (size - k + 1) / 2) + (l - k))] = sqdmat[k,l]
   return d 


@njit(parallel=True, fastmath=False, cache=True)         ####dissimilarity defautl, generalizar para diferentes sigma_0 e epsilon
def dissimilarity(coords, d, k, s0,size, atoms, total):
    epsilon=0.15
    sigma_0=s0
    for l in prange (k, size):
       N=0
       ep=0
       for i in range(atoms):
          for j in range (i+1,atoms):
             xijk = (coords[k][i][0] - coords[k][j][0])
             yijk = (coords[k][i][1] - coords[k][j][1])
             zijk = (coords[k][i][2] - coords[k][j][2])
             rijk = np.sqrt(xijk*xijk + yijk*yijk + zijk*zijk)
             xijl = (coords[l][i][0] - coords[l][j][0])
             yijl = (coords[l][i][1] - coords[l][j][1])
             zijl = (coords[l][i][2] - coords[l][j][2])
             rijl = np.sqrt(xijl*xijl + yijl*yijl + zijl*zijl)
             dr = (rijk-rijl)**2
             sigma = sigma_0*abs(i-j)**epsilon 
             ep = ep + np.exp(-dr/(2*sigma**2))
             N = N + 1
       d[int(total - ((size - k) * (size - k + 1) / 2) + (l - k))] = 1-ep/N
    pass   
       
@njit(parallel=True, fastmath=False, cache=True)         ####dissimilarity defautl, generalizar para diferentes sigma_0 e epsilon
def dissimilarity_no_verbose(coords, d, s0,size, atoms, total):
    epsilon=0.15
    sigma_0=s0
    for k in prange (size):
        for l in range (k, size):
            N=0
            ep=0
            for i in range(atoms):
                for j in range (i+1,atoms):
                    xijk = (coords[k][i][0] - coords[k][j][0])
                    yijk = (coords[k][i][1] - coords[k][j][1])
                    zijk = (coords[k][i][2] - coords[k][j][2])
                    rijk = np.sqrt(xijk*xijk + yijk*yijk + zijk*zijk)
                    xijl = (coords[l][i][0] - coords[l][j][0])
                    yijl = (coords[l][i][1] - coords[l][j][1])
                    zijl = (coords[l][i][2] - coords[l][j][2])
                    rijl = np.sqrt(xijl*xijl + yijl*yijl + zijl*zijl)
                    dr = (rijk-rijl)**2
                    sigma = sigma_0*abs(i-j)**epsilon 
                    ep = ep + np.exp(-dr/(2*sigma**2))
                    N = N + 1
            d[int(total - ((size - k) * (size - k + 1) / 2) + (l - k))] = 1-ep/N
    return d

def calc_dmat(coords, s0, outdm, verbose):
    """
      Calculates the dissimilarity matrix using coordinates of C-alpha carbons.
      Using other representations, such as all atoms or different number of 
      coarse-grain beads, requires adjusting the residue number.
      
      Parameters
      ----------
      coords : np.float32
          C-alpha atomic coordinates in Angstrom [frames, atom_index, (x, y, z)]
      s0 : float
          Gaussian width. Can be changed to gauge the dissimilarity. Default value is one.
    """
    # Optional: Import tqdm if verbose is enabled
    if verbose:
        try:
            from tqdm import tqdm
        except ImportError:
            print("Error: tqdm is not installed. Please install it using `pip install tqdm` to enable verbose mode with progress bar. \n")
            sys.exit(1)  # Disable verbose mode if tqdm is not available  
            
    size, atoms, _=coords.shape
    total = int(size*(size+1)/2)
    dmat=np.zeros(total, dtype=np.float32)  
  
    if verbose:  
        for k in tqdm(range (size), desc='Calculating the dissimilarity matrix'):
            dissimilarity(coords, dmat, k, s0, size, atoms, total)
        
    else:
        dmat = dissimilarity_no_verbose(coords, dmat, s0, size, atoms, total)

    if outdm!=None:
        if os.path.exists(outdm):
            backup_file(outdm)
        np.save(outdm, dmat)
        print("Dissimilarity matrix save in condensed form to the file {} \n".format(outdm))
    return dmat
    
def get_coords(traj_file, top_file):
    # Load the trajectory file
        if traj_file[-4:] != '.pdb' and top_file==None:
            raise RuntimeError("A topology file is needed! \n")
        try:
            if traj_file[-4:] == '.pdb':
                traj = md.load(traj_file)
            else:
                traj=md.load(traj_file, top=top_file)
            print("Trajectory succesfully loaded! \n")
        except:
            raise RuntimeError("Error loading trajectory. \n")
        
        # Select C-alpha and get coordinates in Angstroms
        ca_indices=traj.topology.select("name == CA")
        coords=traj.xyz
        coords=coords[:,ca_indices,:]*10   # Angstrom
        return coords

def backup_file(file_path):
      backup_path = file_path + '.bak'
      shutil.move(file_path, backup_path)
      print(f"Existing file '{file_path}' backed up to '{backup_path}'. \n")
      
def read_dmat(dmat_file):
    try:
        try:
            dmat = np.load(dmat_file)
        except:
            dmat = np.loadtxt(dmat_file)
    except:
        raise RuntimeError("Failed to load the dissimilarity matrix from both binary .npy and text file formats. \n")

    if len(dmat.shape)==1:
        size = int(0.5*(np.sqrt(1+8*len(dmat)-1)))
        if dmat[0]!=0: 
            raise RuntimeError("Make sure the condensed matrix includes the main diagonal zeros. \n")
        print("Loaded a condensed matrix for {} conformations \n".format(size))
    elif len(dmat.shape)==2:
        size, size2 =dmat.shape
        if size!=size2:
            raise RuntimeError("The dissimilarity matrix should be square (NxN) \n")
        # Check if the loaded matrix is approximately symmetric
        is_sym = np.allclose(dmat, dmat.T, atol=1e-3)
        if not is_sym:
            raise RuntimeError("The dissimilarity matrix should be symmetric \n")
        dmat = square_to_condensed(dmat)
        print("Loaded a square matrix for {} conformations \n".format(size))
    else:
        raise RuntimeError('''The dissimilarity matrix should be a symmetrical nxn or its condensed form.
                              See tutorials for details. \n''')
    return dmat, size
  
    
def main():
    """ Read either a trajectory, a dissimilarity matrix or an MDtraj coordinate matrix and output the projection.
           
    Args:
        -f (filename): Pre-processed MD trajectory to be projected in any format recognized by MDtraj
        -t (filename): Topology information (eg. frame.pdb) for loading binary trajectories
        -dm (filename): File containing a dissimilarity matrix (default: None)
        -c (filename): File containing the matrix coords.npy. Coordinates should be in Angstrom (default: None).
        -s0 (float): sigma_0 for dissimilarity (default: 1).
        -lr0 (float): Initial value for the learning rate (default: 0.4)
        -lrmin (lrmin): Minimum value for the learning rate (default: 0.05)
        -d (float): Exponential decay of the learning_rate (default: 0.95)
        -it (int): Maximum number of iterations (default: sqrt(n_frames))
        -tol (float): Residual error tolerance to stop iterations (default: 0)
        -o (filename): Name to save the ELViM projection coordinates file (default: projection.out)
        -odm (filename): Name to save the dissimilarity matrix (default: None)
        - v (--verbose) : Show progress bar (requires the tqdm library)
    Returns:
        output (file): binary numpy file (.npy) containing the compressed dissimilarity matrix (default: coords.npy)
        output (file): ELViM projection coordinates file (default: projection.out)
    """
    
    message='''This program can:
        1. Read the trajectory file (pdb or other formats +ref.pdb, calculate the dissimilarity matrix
           and perform the ELViM projection. \n
        2. Read a distance matrix in the symmetrical square or condensed format and perform the
           ELViM projection.
        3. Read a binary file containing a coordinate matrix saved in MDtraj format in angstroms (traj.xyz*10)    
        * Only one of the previous options should be provided.
        * To save the dissimilarity matrix for other projections use -odm name.npy'''
        
    texto = """
       _______   ___       ___      ___ ___  _____ ______      
      |\  ___ \ |\  \     |\  \    /  /|\  \|\   _ \  _   \    
      \ \   __/|\ \  \    \ \  \  /  / | \  \ \  \\\__\ \  \   
       \ \  \_|/_\ \  \    \ \  \/  / / \ \  \ \  \\|__| \  \  
        \ \  \_|\ \ \  \____\ \    / /   \ \  \ \  \    \ \  \ 
         \ \_______\ \_______\ \__/ /     \ \__\ \__\    \ \__\\
          \|_______|\|_______|\|__|/       \|__|\|__|     \|__|
      
                               ELViM :)
                             Version 1.1
      
      
      Please cite:

      Viegas, R. G., Martins, I. B., Sanches, M. N., Oliveira Junior, A. B., 
      Camargo, J. B. D., Paulovich, F. V., & Leite, V. B. (2024). 
      ELViM: Exploring Biomolecular Energy Landscapes through Multidimensional Visualization. 
      Journal of Chemical Information and Modeling, 64(8), 3443-3450.
      
      Oliveira Jr, A. B., Yang, H., Whitford, P. C., & Leite, V. B. (2019). 
      Distinguishing biomolecular pathways and metastable states. 
      Journal of Chemical Theory and Computation, 15(11), 6482-6490.
      
  """

    print(texto)
        
    parser = argparse.ArgumentParser(description=message)
    parser.add_argument("-f", dest="trajectory_file", 
                        action="store", type=str, default=None,
                        help="Name of the trajectory file (in any format recognized by MDtraj)")
    parser.add_argument("-t", dest="topology_file", 
                        action="store", type=str, default=None,
                        help="Name of the topology file (eg. 'frame.pdb', necessary for binary trajectories)")
    parser.add_argument("-dm", dest="dmat", 
                  action="store", type=str, default=None,
                  help="Name of a precomputed dissimilarity matrix file (default: None).")
    parser.add_argument("-c", dest="coords", 
                  action="store", type=str, default=None,
                  help="Name of the coordinate matrix file in MDtraj format (default: None).")
    parser.add_argument("-s0", dest="sigma0", 
                        action="store", type=float, default=1.0,
                        help=" sigma_0 for dissimilarity (default: 1).")
    parser.add_argument("-lr0", dest="lr0", action="store", type=float, default=0.4,
                  help="Initial learning rate (default: 0.4).")
    parser.add_argument("-lrmin", dest="lrmin", action="store", type=float, default=0.05,
                  help="Minimum value for learning rate (default: 0.05).")
    parser.add_argument("-d", dest="decay", action="store", type=float, default=0.95,
                  help="Learning rate decay exponent (default: 0.95)")
    parser.add_argument("-it", dest="max_it", 
                  action="store", type=int, default=None,
                  help="Maximum number of iteration (default sqrt(n_frames))")
    parser.add_argument("-tol", dest="tolerance", action="store", type=float, default=0,
                  help="tolerance to achieve convergence (default: 0)")
    parser.add_argument("-o", dest="output_proj",
                  action="store", type=str, default="projection.out",
                  help="ELViM projection coordinates (default: projection.out)")
    parser.add_argument("-odm", dest="output_dmat",
                        action="store", type=str, default=None,
                        help="Condensed dissimilarity matrix in binary file (default: None).")
    parser.add_argument('-v', '--verbose', action='store_true', help="Show progress information (require tqdm library)")
   

   
    args = parser.parse_args()
    traj_file=args.trajectory_file
    top_file=args.topology_file
    dmat_file=args.dmat
    coords_file=args.coords
    s0 = args.sigma0
    max_it=args.max_it
    lr0=args.lr0
    lrmin=args.lrmin
    tol=args.tolerance
    decay=args.decay
    outdm=args.output_dmat
    outp=args.output_proj
    verbose=args.verbose
    

#### Check if only one of the possible files are parsed to avoid ambiguities between them.    
    files = [traj_file, dmat_file, coords_file]
    nfiles = sum(files is not None for files in files)
    if nfiles == 0:
        raise RuntimeError('''One of the following files should be provided:
                              (1) A trajectory file (plus a topology)
                              (2) A dissimilarity matrix 
                              (3) A MDtraj coordinates file
                              Try "python ELViM.py -h" to see the options. \n''')
    elif nfiles > 1:
        raise RuntimeError('''Only one of the following files should be provided:
                              (1) A trajectory file (plus a topology)
                              (2) A dissimilarity matrix 
                              (3) An MDtraj coordinates file). \n''')


     #Read traj and calculate the dissimilarity matrix                         
    if traj_file!=None:                       
       coords=get_coords(traj_file, top_file)
       dmat = calc_dmat(coords, s0, outdm, verbose)
       size, atom, _=coords.shape
       print('Dissimilaty matrix calculated considering {} conformations and {} alpha carbons \n'.format(size, atom))

    elif dmat_file!=None:
       dmat, size = read_dmat(dmat_file)
       print('Dissimilaty matrix loaded. There are {} conformations \n'.format(size))
       
    elif coords_file!=None:
       try:
            coords = np.load(coords_file)
            if coords.shape[2]!=3:
                raise RuntimeError("Dimension error: the coordinate array should follow MDtraj format in angstroms (trajectory.xyz*10) ")
            print(''''
                  Coordinates matix loaded. Found {} frames and {} atoms.
                  Make sure coordinates contain only CA atoms and are expressed in angstroms \n.
                  '''. format(coords.shape[0], coords.shape[1]))        
       except:
            raise RuntimeError('''Failed to load the coordinates file. 
                                  Make sure it is a binary numpy file (.npy),
                                  and the matrix follows the MDtraj format (trajectory.xyz*10) \n''')
       dmat = calc_dmat(coords, s0, outdm, verbose)
       size, atom, _=coords.shape
       print('Dissimilaty matrix calculated considering {} conformations and {} alpha carbons \n'.format(size, atom)) 
  
    ### Make the projection
    ### Initializing the projection with points randomly distributed
    projection = np.random.random((size, 2))   
    if max_it==None:
        max_it = int(np.sqrt(size))
    
    ### Running force scheme
    nr_iterations, error, kstress = force.execute(dmat, projection, max_it, verbose, lr0, lrmin, decay, tol)
    print('ELViM projection done with {} iteratons. \n'.format(max_it))
    ### write the projection coordinates file
    if os.path.exists(outp):
       backup_file(outp)
    with open(outp, 'w') as f:
       f.write("# ELViM Coordinates file. \n") 
       if dmat_file==None:
           f.write("# Dissimilarity matrix calculated with sigma0={} \n".format(s0))
       else:
           f.write("# Dissimilarity matrix read from the file : {}\n".format(dmat_file))
       f.write("# {} iterations with lr0 = {:5.3f}, lrmin = {:5.3f}, and decay = {:5.3f} \n".format(nr_iterations, lr0, lrmin, decay))
       f.write("# Mean projection error = {:6.4f}, Stress = {:6.4f} \n".format(error[nr_iterations-1], kstress[nr_iterations-1]))
       np.savetxt(f, projection, delimiter=" ", fmt="%.4f")
    
if __name__=="__main__":
	main()
	exit(0)

{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "900f6f7d-7efa-4bdb-a667-f8578f013799",
   "metadata": {
    "tags": []
   },
   "source": [
    "# Testing Dose Calculator Function\n",
    "I'll do initial testing stuff and miscellaneous things here and then add them into `dose_calculator_tests.py`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "5869feba-5ea0-48bf-93af-26a8f32eed41",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Populating the interactive namespace from numpy and matplotlib\n"
     ]
    }
   ],
   "source": [
    "%pylab ipympl \n",
    "%load_ext autoreload\n",
    "import siddon as sd\n",
    "import dose_calculator as dc\n",
    "import imshow_slider as ims\n",
    "from topas2numpy import BinnedResult\n",
    "import pickle\n",
    "from scipy import interpolate\n",
    "import spekpy as sp"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e1b15e18-7a5c-4b26-ad2a-8b453873abe9",
   "metadata": {
    "tags": []
   },
   "source": [
    "## Testing Dose One\n",
    "This is a 120 keV monoenergetic pencil beam in an ellipse shape moving from positive to negative z. The medium is a 5cm x 5cm x 5cm block with 50 voxels in each axis."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "8529847d-bbd1-498c-9190-8fe7d195c38f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Calling Superposition\n",
      "Kernel Interpolated\n",
      "0\n",
      "1\n",
      "2\n",
      "3\n",
      "CPU times: user 2min 20s, sys: 21.6 s, total: 2min 42s\n",
      "Wall time: 7min 20s\n"
     ]
    }
   ],
   "source": [
    "%%time \n",
    "\n",
    "# for a CT array of (Nx-1,Ny-1,Nz-1) voxels\n",
    "Nx = 51\n",
    "Ny = 51\n",
    "Nz = 51\n",
    "\n",
    "# distances between the x,y,z planes (also the lengths of the sides of the voxels) in cm\n",
    "dx = 0.1\n",
    "dy = 0.1\n",
    "dz = 0.1\n",
    "\n",
    "# initial and final coordinates of the beam\n",
    "x1,x2 = (0,0)\n",
    "y1,y2 = (0,0)\n",
    "z1,z2 = (2.5,-2.5)\n",
    "\n",
    "# adjustment from center \n",
    "adjust = 0.025\n",
    "\n",
    "# initial plane coordinates\n",
    "xplane1 = -2.5\n",
    "yplane1 = -2.5\n",
    "zplane1 = -2.5\n",
    "\n",
    "# beam info and filename\n",
    "beam_energy = 0.120 # in MeV\n",
    "fluence_0 = 3.183098862 * 10**10 # photon/cm^2\n",
    "# fluence_0 = 9.93 * 10**8 # photon/cm^2\n",
    "filenames = ['energy_absorption_coeffs.txt','energy_absorption_coeffs_cort_bone.txt']\n",
    "\n",
    "# kernel info\n",
    "kernelname_water = '../Topas/Kernels/WaterKernel13.csv'\n",
    "kernelname_bone = '../Topas/Kernels/BoneKernel3.csv'\n",
    "kernelnames = [kernelname_water,kernelname_bone]\n",
    "# kernelname = '../Topas/RealKernel1.csv'\n",
    "kernel_size = (4,4,4) # cm \n",
    "\n",
    "# effective distance from center of kernel \n",
    "eff_dist = (2,2,2) # cm\n",
    "\n",
    "# making materials array \n",
    "materials = []\n",
    "for i in range(Nx-1):\n",
    "    materials.append([])\n",
    "    for j in range(Ny-1):\n",
    "        materials[i].append([])\n",
    "        for k in range(Nz-1):\n",
    "            materials[i][j].append('w')\n",
    "\n",
    "# number of cores to use\n",
    "num_cores = 16\n",
    "\n",
    "# Siddon((Nx,Ny,Nz),(dx,dy,dz),((x1,x2),(y1,y2),(z1,z2)),(xplane1,yplane1,zplane1),plot=True)\n",
    "# terma = sd.TERMA((Nx,Ny,Nz),(dx,dy,dz),((x1,x2),(y1,y2),(z1,z2)),(xplane1,yplane1,zplane1),[beam_energy],[fluence_0],mu)\n",
    "\n",
    "dose = dc.Dose_Calculator((Nx,Ny,Nz),(dx,dy,dz),[((x1+adjust,x2+adjust),(y1+adjust,y2+adjust),(z1,z2)),((x1-adjust,x2-adjust),(y1+adjust,y2+adjust),(z1,z2)),((x1+adjust,x2+adjust),(y1-adjust,y2-adjust),(z1,z2)),((x1-adjust,x2-adjust),(y1-adjust,y2-adjust),(z1,z2))],(xplane1,yplane1,zplane1),[beam_energy],[fluence_0],filenames,kernelnames,kernel_size,eff_dist,materials,num_cores)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "75a1ccfb-0144-47f8-9892-088d61e77c28",
   "metadata": {},
   "outputs": [],
   "source": [
    "with 16 \"cores\":\n",
    "CPU times: user 9.09 s, sys: 1.1 s, total: 10.2 s\n",
    "Wall time: 13min 36s"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "ae4c066d-cd48-4345-ab62-3c6115cd581c",
   "metadata": {},
   "outputs": [],
   "source": [
    "pickle.dump(dose,open('dose_38.pickle','wb'))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2fbc050a-8bd1-4873-a7b0-4da4f461bf88",
   "metadata": {},
   "source": [
    "## Testing Dose Two\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "6e64aef1-c300-491b-9f0a-e7e65064ac4f",
   "metadata": {},
   "outputs": [],
   "source": [
    "s = sp.Spek(kvp=120,th=12) # Generate a spectrum (80 kV, 12 degree tube angle)\n",
    "s.filter('Al', 4.0) # Filter by 4 mm of Al\n",
    "\n",
    "hvl = s.get_hvl1() # Get the 1st HVL in mm Al\n",
    "\n",
    "# print(hvl) # Print out the HVL value (Python3 syntax)\n",
    "\n",
    "beam_energy,fluence_0 = s.get_spectrum()\n",
    "beam_energy = beam_energy/1000"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "0a677621-4bcf-4eb1-abf3-c28082fcfa1e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "238"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# for fluence in fluence_0:\n",
    "#     print(fluence/sum(fluence_0),end=' ')\n",
    "len(beam_energy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "9eaf59d5-eeea-4c5f-bd24-b3667bf8fc27",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CPU times: user 9.15 s, sys: 1.01 s, total: 10.2 s\n",
      "Wall time: 12min 17s\n"
     ]
    }
   ],
   "source": [
    "%%time \n",
    "\n",
    "# for a CT array of (Nx-1,Ny-1,Nz-1) voxels\n",
    "Nx = 51\n",
    "Ny = 51\n",
    "Nz = 51\n",
    "\n",
    "# distances between the x,y,z planes (also the lengths of the sides of the voxels) in cm\n",
    "dx = 0.1\n",
    "dy = 0.1\n",
    "dz = 0.1\n",
    "\n",
    "# initial and final coordinates of the beam\n",
    "x1,x2 = (0,0)\n",
    "y1,y2 = (0,0)\n",
    "z1,z2 = (2.5,-2.5)\n",
    "\n",
    "# adjustment from center \n",
    "adjust = 0.025\n",
    "\n",
    "# initial plane coordinates\n",
    "xplane1 = -2.5\n",
    "yplane1 = -2.5\n",
    "zplane1 = -2.5\n",
    "\n",
    "# beam info and filename\n",
    "filename = 'energy_absorption_coeffs.txt'\n",
    "\n",
    "# kernel info\n",
    "kernelname = '../Topas/RealKernel1.csv'\n",
    "kernel_size = (2,2,2) # cm \n",
    "\n",
    "# effective distance from center of kernel \n",
    "eff_dist = (0.1,0.1,0.1) # cm\n",
    "\n",
    "# number of cores to use\n",
    "num_cores = 16\n",
    "\n",
    "# Siddon((Nx,Ny,Nz),(dx,dy,dz),((x1,x2),(y1,y2),(z1,z2)),(xplane1,yplane1,zplane1),plot=True)\n",
    "# terma = sd.TERMA((Nx,Ny,Nz),(dx,dy,dz),((x1,x2),(y1,y2),(z1,z2)),(xplane1,yplane1,zplane1),[beam_energy],[fluence_0],mu)\n",
    "\n",
    "dose = sd.Dose_Calculator((Nx,Ny,Nz),(dx,dy,dz),[((x1+adjust,x2+adjust),(y1+adjust,y2+adjust),(z1,z2)),((x1-adjust,x2-adjust),(y1+adjust,y2+adjust),(z1,z2)),((x1+adjust,x2+adjust),(y1-adjust,y2-adjust),(z1,z2)),((x1-adjust,x2-adjust),(y1-adjust,y2-adjust),(z1,z2))],(xplane1,yplane1,zplane1),beam_energy,fluence_0,filename,kernelname,kernel_size,eff_dist,num_cores)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "027a44a4-ca2a-4f88-b452-004df5f22586",
   "metadata": {},
   "outputs": [],
   "source": [
    "pickle.dump(dose,open('dosetest2_1.pickle','wb'))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a40052eb-acf1-4ca6-8f93-8505641f35f2",
   "metadata": {},
   "source": [
    "## Testing Dose Three"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "8a9178c4-ff1e-43f1-8652-42a3ec55bac4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Calling Superposition\n",
      "Kernel Interpolated\n",
      "0\n",
      "1\n",
      "2\n",
      "3\n",
      "CPU times: user 2min 22s, sys: 23.5 s, total: 2min 45s\n",
      "Wall time: 7min 57s\n"
     ]
    }
   ],
   "source": [
    "%%time \n",
    "\n",
    "# for a CT array of (Nx-1,Ny-1,Nz-1) voxels\n",
    "Nx = 51\n",
    "Ny = 51\n",
    "Nz = 51\n",
    "\n",
    "# distances between the x,y,z planes (also the lengths of the sides of the voxels) in cm\n",
    "dx = 0.1\n",
    "dy = 0.1\n",
    "dz = 0.1\n",
    "\n",
    "# initial and final coordinates of the beam\n",
    "x1,x2 = (0,0)\n",
    "y1,y2 = (0,0)\n",
    "z1,z2 = (2.5,-2.5)\n",
    "\n",
    "# adjustment from center \n",
    "adjust = 0.025\n",
    "\n",
    "# initial plane coordinates\n",
    "xplane1 = -2.5\n",
    "yplane1 = -2.5\n",
    "zplane1 = -2.5\n",
    "\n",
    "# beam info and filename\n",
    "beam_energy = 0.120 # in MeV\n",
    "# fluence_0 = 3.183098862 * 10**8 # photon/cm^2\n",
    "# fluence_0 = 2.53 * 10**8 # photon/cm^2\n",
    "fluence_0 = 9.93 * 10**8 # photon/cm^2\n",
    "# filename = 'energy_absorption_coeff.txt'\n",
    "filenames = ['energy_absorption_coeffs.txt','energy_absorption_coeffs_cort_bone.txt']\n",
    "\n",
    "# kernel info\n",
    "kernelname_water = '../Topas/Kernels/WaterKernel13.csv'\n",
    "kernelname_bone = '../Topas/Kernels/BoneKernel3.csv'\n",
    "kernelnames = [kernelname_water,kernelname_bone]\n",
    "# kernelname = '../Topas/RealKernel1.csv'\n",
    "kernel_size = (4,4,4) # cm \n",
    "\n",
    "# effective distance from center of kernel \n",
    "eff_dist = (2,2,2) # cm\n",
    "\n",
    "# making materials array \n",
    "materials = []\n",
    "for i in range(Nx-1):\n",
    "    materials.append([])\n",
    "    for j in range(Ny-1):\n",
    "        materials[i].append([])\n",
    "        for k in range((Nz-1)//2):\n",
    "            materials[i][j].append('b')\n",
    "        for k in range((Nz-1)//2,Nz-1):\n",
    "            materials[i][j].append('w')\n",
    "\n",
    "# number of cores to use\n",
    "num_cores = 16\n",
    "\n",
    "# Siddon((Nx,Ny,Nz),(dx,dy,dz),((x1,x2),(y1,y2),(z1,z2)),(xplane1,yplane1,zplane1),plot=True)\n",
    "# terma = sd.TERMA((Nx,Ny,Nz),(dx,dy,dz),((x1,x2),(y1,y2),(z1,z2)),(xplane1,yplane1,zplane1),[beam_energy],[fluence_0],mu)\n",
    "\n",
    "dose = dc.Dose_Calculator((Nx,Ny,Nz),(dx,dy,dz),[((x1+adjust,x2+adjust),(y1+adjust,y2+adjust),(z1,z2)),((x1-adjust,x2-adjust),(y1+adjust,y2+adjust),(z1,z2)),((x1+adjust,x2+adjust),(y1-adjust,y2-adjust),(z1,z2)),((x1-adjust,x2-adjust),(y1-adjust,y2-adjust),(z1,z2))],(xplane1,yplane1,zplane1),[beam_energy],[fluence_0],filenames,kernelnames,kernel_size,eff_dist,materials,num_cores)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "76d00927-745f-4687-8f33-988424fa4ffd",
   "metadata": {},
   "outputs": [],
   "source": [
    "pickle.dump(dose,open('dose3_4.pickle','wb'))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "63cba315-892e-4d38-a07d-45074d7a5255",
   "metadata": {},
   "source": [
    "## Testing Dose Four"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "8a3a9df4-1fbf-4c08-8fe8-2ccc743b7bb6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Calling Superposition\n",
      "Kernel Interpolated\n",
      "0\n",
      "1\n",
      "2\n",
      "3\n",
      "4\n",
      "5\n",
      "6\n",
      "7\n",
      "8\n",
      "9\n",
      "10\n",
      "11\n",
      "12\n",
      "13\n",
      "14\n",
      "15\n",
      "16\n",
      "17\n",
      "18\n",
      "19\n",
      "20\n",
      "21\n",
      "22\n",
      "23\n",
      "24\n",
      "25\n",
      "26\n",
      "27\n",
      "28\n",
      "29\n",
      "30\n",
      "31\n",
      "32\n",
      "33\n",
      "34\n",
      "35\n",
      "36\n",
      "37\n",
      "38\n",
      "39\n",
      "40\n",
      "41\n",
      "42\n",
      "43\n",
      "44\n",
      "45\n",
      "46\n",
      "47\n",
      "48\n",
      "49\n",
      "CPU times: user 3min 30s, sys: 40.4 s, total: 4min 11s\n",
      "Wall time: 20min 56s\n"
     ]
    }
   ],
   "source": [
    "%%time \n",
    "\n",
    "# for a CT array of (Nx-1,Ny-1,Nz-1) voxels\n",
    "Nx = 51\n",
    "Ny = 51\n",
    "Nz = 51\n",
    "\n",
    "# distances between the x,y,z planes (also the lengths of the sides of the voxels) in cm\n",
    "dx = 0.1\n",
    "dy = 0.1\n",
    "dz = 0.1\n",
    "\n",
    "# initial and final coordinates of the beam\n",
    "x1,x2 = (0,0)\n",
    "y1,y2 = (0,0)\n",
    "z1,z2 = (2.5,-2.5)\n",
    "\n",
    "# number of rays \n",
    "num_rays = 10\n",
    "\n",
    "rays = dc.MakeFanBeamRays(num_rays,np.pi/6,((x1,x2),(y1,y2),(z1,z2)),direction='y',adjust=0.025)\n",
    "\n",
    "# adjustment from center \n",
    "adjust = 0.01\n",
    "\n",
    "# initial plane coordinates\n",
    "xplane1 = -2.5\n",
    "yplane1 = -2.5\n",
    "zplane1 = -2.5\n",
    "\n",
    "# beam info and filename\n",
    "beam_energy = 0.120 # in MeV\n",
    "# fluence_0 = 3.183098862 * 10**8 # photon/cm^2\n",
    "# fluence_0 = 2.53 * 10**8 # photon/cm^2\n",
    "fluence_0 = 9.93 * 10**8 # photon/cm^2\n",
    "# filename = 'energy_absorption_coeff.txt'\n",
    "filenames = ['energy_absorption_coeffs.txt','energy_absorption_coeffs_cort_bone.txt']\n",
    "\n",
    "# kernel info\n",
    "kernelname_water = '../Topas/Kernels/WaterKernel13.csv'\n",
    "kernelname_bone = '../Topas/Kernels/BoneKernel3.csv'\n",
    "kernelnames = [kernelname_water,kernelname_bone]\n",
    "kernel_size = (4,4,4) # cm \n",
    "\n",
    "# effective distance from center of kernel \n",
    "eff_dist = (1,1,1) # cm\n",
    "\n",
    "# making materials array \n",
    "materials = []\n",
    "for i in range(Nx-1):\n",
    "    materials.append([])\n",
    "    for j in range(Ny-1):\n",
    "        materials[i].append([])\n",
    "        for k in range(Nz-1):\n",
    "            materials[i][j].append('w')\n",
    "\n",
    "# number of cores to use\n",
    "num_cores = 16\n",
    "\n",
    "# Siddon((Nx,Ny,Nz),(dx,dy,dz),((x1,x2),(y1,y2),(z1,z2)),(xplane1,yplane1,zplane1),plot=True)\n",
    "# terma = sd.TERMA((Nx,Ny,Nz),(dx,dy,dz),((x1,x2),(y1,y2),(z1,z2)),(xplane1,yplane1,zplane1),[beam_energy],[fluence_0],mu)\n",
    "\n",
    "dose = dc.Dose_Calculator((Nx,Ny,Nz),(dx,dy,dz),rays,(xplane1,yplane1,zplane1),[beam_energy],[fluence_0],filenames,kernelnames,kernel_size,eff_dist,materials,num_cores)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5c012202-8444-40db-b450-7f714832f87c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# with previous code (slow), 16 cores, and 10 rays\n",
    "CPU times: user 10.5 s, sys: 2.04 s, total: 12.6 s\n",
    "Wall time: 29min 31s"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "c31f7a6e-9cc8-448c-a747-627bd05212b0",
   "metadata": {},
   "outputs": [],
   "source": [
    "pickle.dump(dose,open('dose4_11.pickle','wb'))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "726bc5c0-4183-4499-827c-0b4582ee1dc8",
   "metadata": {},
   "source": [
    "## Testing Dose Five \n",
    "- like TestingDose1 except bone!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "680fe950-6bfb-4998-bbe1-b59bccad1add",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Calling Superposition\n",
      "Kernel Interpolated\n"
     ]
    },
    {
     "ename": "KeyboardInterrupt",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
      "\u001b[0;32m<timed exec>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n",
      "\u001b[0;32m~/NSERC-Summer-2022/Python/dose_calculator.py\u001b[0m in \u001b[0;36mDose_Calculator\u001b[0;34m(num_planes, voxel_lengths, beam_coor, ini_planes, beam_energy, ini_fluence, filenames, kernelnames, kernel_size, eff_distance, mat_array, num_cores)\u001b[0m\n\u001b[1;32m    698\u001b[0m     \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'Calling Superposition'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    699\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 700\u001b[0;31m     \u001b[0menergy_deposit\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mSuperposition\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mkernel_arrays\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mkernel_size\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mnum_planes\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mvoxel_lengths\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mvoxel_info\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mbeam_coor\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0meff_distance\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mmat_array\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mnum_cores\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    701\u001b[0m     \u001b[0;32mreturn\u001b[0m \u001b[0menergy_deposit\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    702\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/NSERC-Summer-2022/Python/dose_calculator.py\u001b[0m in \u001b[0;36mSuperposition\u001b[0;34m(kernel_arrays, kernel_size, num_planes, voxel_lengths, voxel_info, beam_coor, eff_distance, mat_array, num_cores)\u001b[0m\n\u001b[1;32m    489\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    490\u001b[0m     \u001b[0;31m# this is where I can lower size of data too\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 491\u001b[0;31m     \u001b[0mvoxel_array\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0marray\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mx\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0my\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mz\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0mx\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mx_voxels\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0my\u001b[0m \u001b[0;32min\u001b[0m \u001b[0my_voxels\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0mz\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mz_voxels\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    492\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    493\u001b[0m     \u001b[0menergy_deposit\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m: "
     ]
    }
   ],
   "source": [
    "%%time \n",
    "\n",
    "# for a CT array of (Nx-1,Ny-1,Nz-1) voxels\n",
    "Nx = 51\n",
    "Ny = 51\n",
    "Nz = 51\n",
    "\n",
    "# distances between the x,y,z planes (also the lengths of the sides of the voxels) in cm\n",
    "dx = 0.1\n",
    "dy = 0.1\n",
    "dz = 0.1\n",
    "\n",
    "# initial and final coordinates of the beam\n",
    "x1,x2 = (0,0)\n",
    "y1,y2 = (0,0)\n",
    "z1,z2 = (2.5,-2.5)\n",
    "\n",
    "# adjustment from center \n",
    "adjust = 0.025\n",
    "\n",
    "# initial plane coordinates\n",
    "xplane1 = -2.5\n",
    "yplane1 = -2.5\n",
    "zplane1 = -2.5\n",
    "\n",
    "# beam info and filename\n",
    "beam_energy = 0.120 # in MeV\n",
    "# fluence_0 = 3.183098862 * 10**8 # photon/cm^2\n",
    "# fluence_0 = 2.53 * 10**8 # photon/cm^2\n",
    "fluence_0 = 9.93 * 10**8 # photon/cm^2\n",
    "# filename = 'energy_absorption_coeff.txt'\n",
    "filenames = ['energy_absorption_coeffs.txt','energy_absorption_coeffs_cort_bone.txt']\n",
    "\n",
    "# kernel info\n",
    "kernelname_water = '../Topas/Kernels/WaterKernel13.csv'\n",
    "kernelname_bone = '../Topas/Kernels/BoneKernel3.csv'\n",
    "kernelnames = [kernelname_water,kernelname_bone]\n",
    "# kernelname = '../Topas/RealKernel1.csv'\n",
    "kernel_size = (4,4,4) # cm \n",
    "\n",
    "# effective distance from center of kernel \n",
    "eff_dist = (2,2,2) # cm\n",
    "\n",
    "# making materials array \n",
    "materials = []\n",
    "for i in range(Nx-1):\n",
    "    materials.append([])\n",
    "    for j in range(Ny-1):\n",
    "        materials[i].append([])\n",
    "        for k in range(Nz-1):\n",
    "            materials[i][j].append('b')\n",
    "\n",
    "# number of cores to use\n",
    "num_cores = 16\n",
    "\n",
    "# Siddon((Nx,Ny,Nz),(dx,dy,dz),((x1,x2),(y1,y2),(z1,z2)),(xplane1,yplane1,zplane1),plot=True)\n",
    "# terma = sd.TERMA((Nx,Ny,Nz),(dx,dy,dz),((x1,x2),(y1,y2),(z1,z2)),(xplane1,yplane1,zplane1),[beam_energy],[fluence_0],mu)\n",
    "\n",
    "dose = dc.Dose_Calculator((Nx,Ny,Nz),(dx,dy,dz),[((x1+adjust,x2+adjust),(y1+adjust,y2+adjust),(z1,z2)),((x1-adjust,x2-adjust),(y1+adjust,y2+adjust),(z1,z2)),((x1+adjust,x2+adjust),(y1-adjust,y2-adjust),(z1,z2)),((x1-adjust,x2-adjust),(y1-adjust,y2-adjust),(z1,z2))],(xplane1,yplane1,zplane1),[beam_energy],[fluence_0],filenames,kernelnames,kernel_size,eff_dist,materials,num_cores)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "80dc2f97-6c97-4a40-b452-3d32f15f3688",
   "metadata": {},
   "outputs": [],
   "source": [
    "pickle.dump(dose,open('dose5_3pickle','wb'))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d47a5625-f6f3-4920-a6fa-2c6f5fca6cff",
   "metadata": {
    "tags": []
   },
   "source": [
    "## Testing Dose Six\n",
    "This is a 120 keV monoenergetic pencil beam in an ellipse shape moving from positive to negative z. The medium is a 5cm x 5cm x 5cm block with 50 voxels in each axis."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "7ee9cfa4-02eb-4cfe-8ef2-602958e02816",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Calling Superposition\n",
      "Kernel Interpolated\n",
      "0\n",
      "1\n",
      "CPU times: user 1min 19s, sys: 12.9 s, total: 1min 32s\n",
      "Wall time: 4min 34s\n"
     ]
    }
   ],
   "source": [
    "%%time \n",
    "\n",
    "# for a CT array of (Nx-1,Ny-1,Nz-1) voxels\n",
    "Nx = 51\n",
    "Ny = 51\n",
    "Nz = 51\n",
    "\n",
    "# distances between the x,y,z planes (also the lengths of the sides of the voxels) in cm\n",
    "dx = 0.1\n",
    "dy = 0.1\n",
    "dz = 0.1\n",
    "\n",
    "# initial and final coordinates of the beam\n",
    "x1,x2 = (0,0)\n",
    "y1,y2 = (0,tan(pi/6)*5)\n",
    "z1,z2 = (2.5,-2.5)\n",
    "\n",
    "# adjustment from center \n",
    "adjust = 0.025\n",
    "\n",
    "# initial plane coordinates\n",
    "xplane1 = -2.5\n",
    "yplane1 = -2.5\n",
    "zplane1 = -2.5\n",
    "\n",
    "# beam info and filename\n",
    "beam_energy = 0.120 # in MeV\n",
    "fluence_0 = 3.183098862 * 10**10 # photon/cm^2\n",
    "# fluence_0 = 9.93 * 10**8 # photon/cm^2\n",
    "filenames = ['energy_absorption_coeffs.txt','energy_absorption_coeffs_cort_bone.txt']\n",
    "\n",
    "# kernel info\n",
    "kernelname_water = '../Topas/Kernels/WaterKernel13.csv'\n",
    "kernelname_bone = '../Topas/Kernels/BoneKernel3.csv'\n",
    "kernelnames = [kernelname_water,kernelname_bone]\n",
    "# kernelname = '../Topas/RealKernel1.csv'\n",
    "kernel_size = (4,4,4) # cm \n",
    "\n",
    "# effective distance from center of kernel \n",
    "eff_dist = (2,2,2) # cm\n",
    "\n",
    "# making materials array \n",
    "materials = []\n",
    "for i in range(Nx-1):\n",
    "    materials.append([])\n",
    "    for j in range(Ny-1):\n",
    "        materials[i].append([])\n",
    "        for k in range(Nz-1):\n",
    "            materials[i][j].append('w')\n",
    "\n",
    "# number of cores to use\n",
    "num_cores = 16\n",
    "\n",
    "# Siddon((Nx,Ny,Nz),(dx,dy,dz),((x1,x2),(y1,y2),(z1,z2)),(xplane1,yplane1,zplane1),plot=True)\n",
    "# terma = sd.TERMA((Nx,Ny,Nz),(dx,dy,dz),((x1,x2),(y1,y2),(z1,z2)),(xplane1,yplane1,zplane1),[beam_energy],[fluence_0],mu)\n",
    "\n",
    "dose = dc.Dose_Calculator((Nx,Ny,Nz),(dx,dy,dz),[((x1+adjust,x2+adjust),(y1,y2),(z1,z2)),((x1-adjust,x2-adjust),(y1,y2),(z1,z2))],(xplane1,yplane1,zplane1),[beam_energy],[fluence_0],filenames,kernelnames,kernel_size,eff_dist,materials,num_cores)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "6663ebfd-5b4a-438c-a5b2-c307180fe540",
   "metadata": {},
   "outputs": [],
   "source": [
    "pickle.dump(dose,open('dose6_2.pickle','wb'))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a305edf9-7b69-4a89-99b8-2473e714d706",
   "metadata": {
    "tags": []
   },
   "source": [
    "## Reverse Engineering Kernel"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "9cf5960e-6399-4362-b853-d0a473056428",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(3.0, 3.0, 13.0) 0.0\n",
      "(3.0, 3.0, 10.0) 0.0\n",
      "(3.0, 3.0, 12.0) 0.0\n",
      "(3.0, 3.0, 6.0) 6.419899895270212e-07\n",
      "(3.0, 3.0, 11.0) 0.0\n",
      "(3.0, 3.0, 9.0) 0.0\n",
      "(3.0, 3.0, 5.0) 0.0006659781594462842(3.0, 3.0, 3.0) \n",
      "4.8628067551665585e-05\n",
      "(3.0, 3.0, 8.0) 7.422201649956291e-08\n",
      "(3.0, 3.0, 2.0) 3.230064520680644e-07\n",
      "(3.0, 3.0, 7.0) 2.528025660796346e-07\n",
      "(3.0, 3.0, 4.0) 0.0018144549474584369\n",
      "(3.0, 3.0, -1.0) 0.0\n",
      "(3.0, 3.0, 1.0) 6.413210910409552e-08(3.0, 3.0, 0.0)\n",
      " 3.0111572516624414e-08\n",
      "(3.0, 3.0, -2.0) 0.0\n",
      "(3.0, 3.0, -3.0) 0.0\n",
      "(3.0, 3.0, -4.0) 0.0\n",
      "(3.0, 3.0, -5.0) 0.0\n",
      "(4.0, 3.0, 13.0) 0.0\n",
      "(4.0, 3.0, 12.0) 0.0\n",
      "(4.0, 3.0, 8.0) 8.480457472828421e-08\n",
      "(4.0, 3.0, 6.0) 9.194167009919759e-07\n",
      "(4.0, 3.0, 10.0) 0.0\n",
      "(4.0, 3.0, 11.0) 0.0\n",
      "(4.0, 3.0, 7.0) 1.9779083910200444e-07\n",
      "(4.0, 3.0, 3.0) 0.0004282900914994053\n",
      "(4.0, 3.0, 9.0) 0.0\n",
      "(4.0, 3.0, 5.0) 0.012707327037807283\n",
      "(4.0, 3.0, 4.0) 0.0020735952439145592\n",
      "(4.0, 3.0, 1.0) 8.497772198892075e-08\n",
      "(4.0, 3.0, 0.0) 8.833288777891727e-08\n",
      "(4.0, 3.0, 2.0) 2.5793574744466066e-07\n",
      "(4.0, 3.0, -2.0) 0.0\n",
      "(4.0, 3.0, -3.0) 0.0\n",
      "(4.0, 3.0, -1.0) 0.0\n",
      "(4.0, 3.0, -5.0) 0.0\n",
      "(4.0, 3.0, -4.0) 0.0\n",
      "(3.0, 4.0, 12.0) 0.0\n",
      "(3.0, 4.0, 13.0) 0.0\n",
      "(3.0, 4.0, 8.0) 1.3355826394045e-07\n",
      "(3.0, 4.0, 11.0) 0.0\n",
      "(3.0, 4.0, 10.0) 0.0\n",
      "(3.0, 4.0, 7.0) 2.3660722993547786e-07\n",
      "(3.0, 4.0, 5.0) 0.012684661445702511\n",
      "(3.0, 4.0, 9.0) 0.0\n",
      "(3.0, 4.0, 2.0) 3.2092761686273856e-07\n",
      "(3.0, 4.0, -2.0) 0.0\n",
      "(3.0, 4.0, 4.0) 0.0020802265385764214\n",
      "(3.0, 4.0, 1.0) 1.0372132831637139e-07\n",
      "(3.0, 4.0, 0.0) 5.075754388040384e-08\n",
      "(3.0, 4.0, 6.0) 7.680821382484351e-07\n",
      "(3.0, 4.0, 3.0) 0.000432456919845498\n",
      "(3.0, 4.0, -1.0) 0.0\n",
      "(3.0, 4.0, -3.0) 0.0\n",
      "(3.0, 4.0, -4.0) 0.0\n",
      "(3.0, 4.0, -5.0) 0.0\n",
      "(4.0, 4.0, 13.0) 0.0\n",
      "(4.0, 4.0, 12.0) 0.0\n",
      "(4.0, 4.0, 9.0) 0.0\n",
      "(4.0, 4.0, 10.0) 0.0\n",
      "(4.0, 4.0, 7.0) 3.705988960663628e-07\n",
      "(4.0, 4.0, 8.0) 9.892502903799936e-08\n",
      "(4.0, 4.0, 11.0) 0.0\n",
      "(4.0, 4.0, 3.0) 0.0030815566018014065\n",
      "(4.0, 4.0, 0.0) 5.2347737252406056e-08\n",
      "(4.0, 4.0, 5.0) 0.3166433887477645\n",
      "(4.0, 4.0, 6.0) 1.1694043186579824e-06\n",
      "(4.0, 4.0, 2.0) 2.8728604758968235e-07\n",
      "(4.0, 4.0, 4.0) 0.6093311929612528\n",
      "(4.0, 4.0, 1.0) 5.7164979012375356e-08\n",
      "(4.0, 4.0, -1.0) 0.0\n",
      "(4.0, 4.0, -2.0) 0.0\n",
      "(4.0, 4.0, -5.0) 0.0\n",
      "(4.0, 4.0, -3.0) 0.0\n",
      "(4.0, 4.0, -4.0) 0.0\n",
      "CPU times: user 710 ms, sys: 466 ms, total: 1.18 s\n",
      "Wall time: 1min 6s\n"
     ]
    }
   ],
   "source": [
    "%%time \n",
    "\n",
    "# for a CT array of (Nx-1,Ny-1,Nz-1) voxels\n",
    "Nx = 51\n",
    "Ny = 51\n",
    "Nz = 51\n",
    "\n",
    "# distances between the x,y,z planes (also the lengths of the sides of the voxels) in cm\n",
    "dx = 0.1\n",
    "dy = 0.1\n",
    "dz = 0.1\n",
    "\n",
    "# initial and final coordinates of the beam\n",
    "x1,x2 = (0,0)\n",
    "y1,y2 = (0,0)\n",
    "z1,z2 = (2.5,-2.5)\n",
    "\n",
    "# adjustment from center \n",
    "adjust = 0.025\n",
    "\n",
    "# initial plane coordinates\n",
    "xplane1 = -2.5\n",
    "yplane1 = -2.5\n",
    "zplane1 = -2.5\n",
    "\n",
    "# beam info and filename\n",
    "beam_energy = 0.120 # in MeV\n",
    "# fluence_0 = 3.183098862 * 10**8 # photon/cm^2\n",
    "# fluence_0 = 2.53 * 10**8 # photon/cm^2\n",
    "fluence_0 = 9.93 * 10**8 # photon/cm^2\n",
    "# filename = 'energy_absorption_coeff.txt'\n",
    "filename = 'energy_absorption_coeffs.txt'\n",
    "\n",
    "# kernel info\n",
    "kernelname = '../Topas/ReverseEngineerKernel.csv'\n",
    "kernel_size = (0.9,0.9,0.9) # cm \n",
    "\n",
    "# effective distance from center of kernel \n",
    "eff_dist = (1,1,1) # cm\n",
    "\n",
    "# number of cores to use\n",
    "num_cores = 16\n",
    "\n",
    "# Siddon((Nx,Ny,Nz),(dx,dy,dz),((x1,x2),(y1,y2),(z1,z2)),(xplane1,yplane1,zplane1),plot=True)\n",
    "# terma = sd.TERMA((Nx,Ny,Nz),(dx,dy,dz),((x1,x2),(y1,y2),(z1,z2)),(xplane1,yplane1,zplane1),[beam_energy],[fluence_0],mu)\n",
    "\n",
    "dose = dc.Dose_Calculator((Nx,Ny,Nz),(dx,dy,dz),[((x1+adjust,x2+adjust),(y1+adjust,y2+adjust),(z1,z2)),((x1-adjust,x2-adjust),(y1+adjust,y2+adjust),(z1,z2)),((x1+adjust,x2+adjust),(y1-adjust,y2-adjust),(z1,z2)),((x1-adjust,x2-adjust),(y1-adjust,y2-adjust),(z1,z2))],(xplane1,yplane1,zplane1),[beam_energy],[fluence_0],filename,kernelname,kernel_size,eff_dist,num_cores)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "ddb9b8a3-a097-41cd-b631-f0bcb6e79f8a",
   "metadata": {},
   "outputs": [],
   "source": [
    "pickle.dump(dose,open('dose_24.pickle','wb'))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ee10efe1-42dd-473f-a30a-d11afb67a62e",
   "metadata": {},
   "source": [
    "## Other Stuff"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3825bb45-273d-4f09-accd-98927d368f01",
   "metadata": {},
   "outputs": [],
   "source": [
    "x1,x2 = (0,0)\n",
    "y1,y2 = (0,0)\n",
    "z1,z2 = (2.5,-2.5)\n",
    "rays = [((x1,x2),(),(z1,z2)) for ]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "92daa9e5-b1f0-478b-98e8-d4b79219771b",
   "metadata": {},
   "outputs": [],
   "source": [
    "'''\n",
    "making mu interpolation function\n",
    "'''\n",
    "filename = 'energy_absorption_coeffs.txt'\n",
    "\n",
    "coeff_array = np.loadtxt(filename,skiprows=2,dtype=float)\n",
    "\n",
    "# exponentially interpolate \n",
    "mu_linear = interpolate.interp1d(np.log(coeff_array.T[0]),np.log(coeff_array.T[1]),kind='linear',fill_value='extrapolate')\n",
    "mu_l = lambda energy, material: np.exp(mu_linear(np.log(energy))) # CHANGE THIS LATER TO A REAL FUNCTION\n",
    "\n",
    "mu_mass = interpolate.interp1d(np.log(coeff_array.T[0]),np.log(coeff_array.T[2]),kind='linear',fill_value='extrapolate')\n",
    "mu_m = lambda energy, material: np.exp(mu_mass(np.log(energy))) # CHANGE THIS LATER TO A REAL FUNCTION"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "79accc39-e821-4d51-a507-58e92f2dd37d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "bd9249fd26a44edd97a7256fc1a3b7d9",
       "version_major": 2,
       "version_minor": 0
      },
      "image/png": "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",
      "text/html": [
       "\n",
       "            <div style=\"display: inline-block;\">\n",
       "                <div class=\"jupyter-widgets widget-label\" style=\"text-align: center;\">\n",
       "                    Figure\n",
       "                </div>\n",
       "                <img src='data:image/png;base64,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' width=640.0/>\n",
       "            </div>\n",
       "        "
      ],
      "text/plain": [
       "Canvas(toolbar=Toolbar(toolitems=[('Home', 'Reset original view', 'home', 'home'), ('Back', 'Back to previous …"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "x1,x2 = (0,0)\n",
    "y1,y2 = (0,0)\n",
    "z1,z2 = (2.5,-2.5)\n",
    "rays = dc.MakeFanBeamRays(12,np.pi/6,((x1,x2),(y1,y2),(z1,z2)),direction='y',adjust=0.1)\n",
    "\n",
    "ifig=1;close(ifig);figure(ifig)\n",
    "for ray in rays:\n",
    "    plot(ray[2],ray[1])\n",
    "grid(b=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "08997f62-318a-41aa-98a9-9dbd24ca4c9e",
   "metadata": {},
   "source": [
    "#### Testing TERMA with Polyenergetic"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "e688d2de-3c87-46f7-aeba-41bb9ae94fb6",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'mu_l' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m<timed exec>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n",
      "\u001b[0;31mNameError\u001b[0m: name 'mu_l' is not defined"
     ]
    }
   ],
   "source": [
    "%%time\n",
    "# for a CT array of (Nx-1,Ny-1,Nz-1) voxels\n",
    "Nx = 10\n",
    "Ny = 10\n",
    "Nz = 10\n",
    "\n",
    "# distances between the x,y,z planes (also the lengths of the sides of the voxels)\n",
    "dx = 0.1\n",
    "dy = 0.1\n",
    "dz = 0.1\n",
    "\n",
    "# initial and final coordinates of the beam\n",
    "x1,x2 = (0.2,0.5)\n",
    "y1,y2 = (0.4,0.8)\n",
    "z1,z2 = (4,-6)\n",
    "\n",
    "# initial plane coordinates\n",
    "xplane1 = 0\n",
    "yplane1 = 0\n",
    "zplane1 = 0\n",
    "\n",
    "# beam info and filename\n",
    "# beam_energy = 0.120 # in MeV\n",
    "# fluence_0 = 1 # photon/cm^2\n",
    "filename = 'energy_absorption_coeffs.txt'\n",
    "\n",
    "# kernel info\n",
    "kernelname = '../Topas/RealKernel1.csv'\n",
    "kernel_size = (1,1,1) # cm \n",
    "\n",
    "# number of cores to use \n",
    "num_cores = 8\n",
    "\n",
    "# sd.Siddon((Nx,Ny,Nz),(dx,dy,dz),((x1,x2),(y1,y2),(z1,z2)),(xplane1,yplane1,zplane1),plot=True)\n",
    "terma = sd.TERMA((Nx,Ny,Nz),(dx,dy,dz),((x1,x2),(y1,y2),(z1,z2)),(xplane1,yplane1,zplane1),beam_energy,fluence_0,mu_l,mu_m)\n",
    "\n",
    "# dose = sd.Dose_Calculator((Nx,Ny,Nz),(dx,dy,dz),[((x1,x2),(y1,y2),(z1,z2))],(xplane1,yplane1,zplane1),beam_energy,fluence_0,filename,kernelname,kernel_size,num_cores)\n",
    "\n",
    "# sd.TERMA?\n",
    "terma"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "07f3e1b0-8e91-4cfa-b336-3b0b6d1a4d2c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.1613015136737795"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mu_l(0.12,'water')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "fa9f2fee-5e7c-4f07-8ca5-2411eef0d6ea",
   "metadata": {},
   "outputs": [],
   "source": [
    "# for a CT array of (Nx-1,Ny-1,Nz-1) voxels\n",
    "Nx = 16\n",
    "Ny = 16\n",
    "Nz = 16\n",
    "\n",
    "# distances between the x,y,z planes (also the lengths of the sides of the voxels) in cm\n",
    "dx = 0.05\n",
    "dy = 0.05\n",
    "dz = 0.05\n",
    "\n",
    "# initial and final coordinates of the beam\n",
    "x1,x2 = (0,0)\n",
    "y1,y2 = (0,0)\n",
    "z1,z2 = (-0.8,0.8)\n",
    "\n",
    "# initial plane coordinates\n",
    "xplane1 = -0.8\n",
    "yplane1 = -0.8\n",
    "zplane1 = -0.8\n",
    "\n",
    "# beam info and filename\n",
    "beam_energy = 0.120 # in MeV\n",
    "fluence_0 = 3.183098862 * 10**8 # photon/cm^2\n",
    "filename = 'energy_absorption_coeffs.txt'\n",
    "\n",
    "# kernel info\n",
    "kernelname = '../Topas/RealKernel1.csv'\n",
    "kernel_size = (1,1,1) # cm \n",
    "\n",
    "# number of cores to use \n",
    "num_cores = 8\n",
    "\n",
    "# siddon = sd.Siddon((Nx,Ny,Nz),(dx,dy,dz),((x1,x2),(y1,y2),(z1,z2)),(xplane1,yplane1,zplane1))\n",
    "# terma = sd.TERMA((Nx,Ny,Nz),(dx,dy,dz),((x1,x2),(y1,y2),(z1,z2)),(xplane1,yplane1,zplane1),beam_energy,fluence_0,mu)\n",
    "dose = sd.Dose_Calculator((Nx,Ny,Nz),(dx,dy,dz),[((x1,x2),(y1,y2),(z1,z2))],(xplane1,yplane1,zplane1),[beam_energy],[fluence_0],filename,kernelname,kernel_size,num_cores)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "112e1d3c-cc3b-4233-876b-d612551bcd20",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(15, 15, 15)"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "shape(dose)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "3513af4a-0c38-458b-bf41-17151d79cdc4",
   "metadata": {},
   "outputs": [],
   "source": [
    "s = sp.Spek(kvp=120,th=12) # Generate a spectrum (80 kV, 12 degree tube angle)\n",
    "s.filter('Al', 4.0) # Filter by 4 mm of Al\n",
    "\n",
    "hvl = s.get_hvl1() # Get the 1st HVL in mm Al\n",
    "\n",
    "# print(hvl) # Print out the HVL value (Python3 syntax)\n",
    "\n",
    "beam_energy,fluence_0 = s.get_spectrum()\n",
    "\n",
    "# sp.Spek?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "6e9be349-5364-4de1-8f80-8559d71ac587",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CPU times: user 8.05 s, sys: 148 ms, total: 8.2 s\n",
      "Wall time: 8.57 s\n"
     ]
    }
   ],
   "source": [
    "%%time\n",
    "# for a CT array of (Nx-1,Ny-1,Nz-1) voxels\n",
    "Nx = 10\n",
    "Ny = 10\n",
    "Nz = 10\n",
    "\n",
    "# distances between the x,y,z planes (also the lengths of the sides of the voxels)\n",
    "dx = 0.1\n",
    "dy = 0.1\n",
    "dz = 0.1\n",
    "\n",
    "# initial and final coordinates of the beam\n",
    "x1,x2 = (0.2,0.5)\n",
    "y1,y2 = (0.4,0.8)\n",
    "z1,z2 = (4,-6)\n",
    "\n",
    "# initial plane coordinates\n",
    "xplane1 = 0\n",
    "yplane1 = 0\n",
    "zplane1 = 0\n",
    "\n",
    "# beam info and filename\n",
    "beam_energy = 0.120 # in MeV\n",
    "fluence_0 = 1 # photon/cm^2\n",
    "filename = 'energy_absorption_coeffs.txt'\n",
    "\n",
    "# kernel info\n",
    "kernelname = '../Topas/RealKernel1.csv'\n",
    "kernel_size = (2,2,2) # cm \n",
    "\n",
    "# effective distance from center of kernel \n",
    "eff_dist = (0.5,0.5,0.5) # cm\n",
    "\n",
    "# number of cores to use \n",
    "num_cores = 8\n",
    "\n",
    "# sd.Siddon((Nx,Ny,Nz),(dx,dy,dz),((x1,x2),(y1,y2),(z1,z2)),(xplane1,yplane1,zplane1),plot=True)\n",
    "# terma = sd.TERMA((Nx,Ny,Nz),(dx,dy,dz),((x1,x2),(y1,y2),(z1,z2)),(xplane1,yplane1,zplane1),beam_energy,fluence_0,mu)\n",
    "\n",
    "dose = dc.Dose_Calculator((Nx,Ny,Nz),(dx,dy,dz),[((x1,x2),(y1,y2),(z1,z2))],(xplane1,yplane1,zplane1),[beam_energy],[fluence_0],filename,kernelname,kernel_size,eff_dist,num_cores)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5a727dda-bf55-4e07-9123-1063a79addd7",
   "metadata": {},
   "source": [
    "*I'm guessing a lot of this time is coming from interpolating the kernel at the beginning, maybe I'll try to run some tests with bigger arrays*"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2b9c6a88-e1bf-4fcf-a04b-1ce8bd935ea1",
   "metadata": {},
   "source": [
    "with 8 cores:\n",
    "CPU times: user 7.96 s, sys: 151 ms, total: 8.11 s\n",
    "Wall time: 8.55 s"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "923da6f2-ced4-4b29-95fb-4cd36df2cf07",
   "metadata": {},
   "source": [
    "with 4 cores:\n",
    "CPU times: user 7.99 s, sys: 140 ms, total: 8.14 s\n",
    "Wall time: 8.76 s"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "02b14f2e-7f8e-43dc-bf09-c4b698145133",
   "metadata": {},
   "outputs": [],
   "source": [
    "for n in range(len(voxel_info)):\n",
    "    for voxel in voxel_info[n]:\n",
    "        if abs(voxel['d'] - 0.1) > 0.00000001:\n",
    "            print('oops')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "47dff494-51f0-455c-8315-71eed385238c",
   "metadata": {},
   "outputs": [],
   "source": [
    "dose_im_array = []\n",
    "\n",
    "for dose_row in dose:\n",
    "    dose_im_array.append(dose_row['energy'])\n",
    "\n",
    "dose_im_array = np.array(dose_im_array)\n",
    "\n",
    "dose_im_array = dose_im_array.reshape(Nx-1,Ny-1,Nz-1)\n",
    "\n",
    "pickle.dump(dose_im_array,open('dose_test.pickle','wb'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "ea03e31b-387b-45e9-9067-e2d1f0f21d1f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "6"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a = array([1,2])\n",
    "b = []"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "ac41d656-69e1-47ea-b726-4bcf44627fff",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "716.6427428588867"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "interaction_zone = []\n",
    "for j in kernel_array[47:52]:\n",
    "    for k in j[47:52]:\n",
    "        interaction_zone.append(k[47:52])\n",
    "interaction_zone = np.array(interaction_zone)\n",
    "np.mean(interaction_zone)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "172cf36d-7104-47a2-910d-524e765474cc",
   "metadata": {},
   "source": [
    "### Slider to Look at 3D Stuff"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "aba68d76-c685-40c3-ba8c-0fde62a2cd5a",
   "metadata": {},
   "outputs": [],
   "source": [
    "kernelname = '../Topas/RealKernel1.csv'\n",
    "kernel_array = BinnedResult(kernelname).data['Sum']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "4bf9c815-0996-4b4a-ae1d-d7c24d57ca61",
   "metadata": {},
   "outputs": [],
   "source": [
    "kernelname = '../Topas/ReverseEngineerKernel.csv'\n",
    "kernel_array = BinnedResult(kernelname).data['Sum']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "0364c355-a2c4-4d85-9fef-9e6a59677a13",
   "metadata": {},
   "outputs": [],
   "source": [
    "kernelname = '../Topas/Kernels/WaterKernel6.csv'\n",
    "kernel_array = BinnedResult(kernelname).data['Sum']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "8bd5a289-6b2f-4776-8d08-5a6673175ef9",
   "metadata": {},
   "outputs": [],
   "source": [
    "kernelname = '../Topas/BoneKernel1.csv'\n",
    "kernel_array = BinnedResult(kernelname).data['Sum']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "343715ad-3318-4711-bd0d-298a4595a0a0",
   "metadata": {},
   "outputs": [],
   "source": [
    "data = BinnedResult('../Topas/EnergyFluence.csv').data['Sum']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "e70b54d5-802f-4bbb-83f6-ae832103e5c8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(array([25]), array([24]), array([49]))"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "where(data==np.max(data))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "a95f5121-1625-4b9b-ba94-573ca0c005cb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b0e7134057ab4a45b0f65daf69de3855",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "VBox(children=(HBox(children=(Play(value=0, max=98), IntSlider(value=0, description='axis0', max=98, readout=F…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "40f2dcbe05ec4ea99f4090c7970376ba",
       "version_major": 2,
       "version_minor": 0
      },
      "image/png": "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SV9fX31tz5492bt3b7q6usZ7XABgnHkCOAH19PTk3nvvzYMPPpgZM2bU39fX1taW6dOnp62tLR//+MezZs2anH766Wltbc11112Xrq6uN/UJYADg5CYAJ6AtW7YkSS677LJR63fddVc+8pGPJEm+8pWvZNKkSVm+fHmGh4ezdOnSfO1rXxvnSQGARvBzAHlL/BxAgJObnwNYNu8BBAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwALMAtt9ySpqamrFq1qr525MiR9PT0ZNasWTn11FOzfPnyDA4ONm5IAGDcCMAJbufOnfn617+e97znPaPWV69enYceeij3339/+vv7s3///lx11VUNmhIAGE8CcAL77//+76xYsSJ33nlnTjvttPr60NBQ/umf/ilf/vKX88d//MdZuHBh7rrrrvzwhz/Mk08+2cCJAYDxIAAnsJ6enlxxxRXp7u4etb5r164cO3Zs1PqCBQsyf/78bNu2bbzHBADGWXOjB2Bs3Hfffdm9e3d27tz5unMDAwOZOnVqZs6cOWq9vb09AwMDb3i/4eHhDA8P17+u1Wpv67wAwPjxBHAC2rdvX66//vrcc889mTZt2ttyz97e3rS1tdWPefPmvS33BQDGnwCcgHbt2pUDBw7kwgsvTHNzc5qbm9Pf35/Nmzenubk57e3tOXr0aA4ePDjqdYODg+no6HjDe65bty5DQ0P1Y9++fePwnQAAY8FfAU9Al19+eX7yk5+MWvvoRz+aBQsW5LOf/WzmzZuXKVOmpK+vL8uXL0+S7NmzJ3v37k1XV9cb3rOlpSUtLS1jPjsAMPYE4AQ0Y8aMnHfeeaPWTjnllMyaNau+/vGPfzxr1qzJ6aefntbW1lx33XXp6urK+9///kaMDACMIwFYqK985SuZNGlSli9fnuHh4SxdujRf+9rXGj0WADAOmqqqqho9BCefWq2Wtra2XJZlaW6a0uhxAPgtvVIdy+N5MENDQ2ltbW30OIwzHwIBACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAJygXnrppXz4wx/OrFmzMn369Jx//vl56qmn6uerqsqGDRsyZ86cTJ8+Pd3d3XnhhRcaODEAMF4E4AT0y1/+MkuWLMmUKVPyve99L88880y+9KUv5bTTTqtfs2nTpmzevDlbt27N9u3bc8opp2Tp0qU5cuRIAycHAMZDc6MH4O23cePGzJs3L3fddVd9rbOzs/7rqqpy66235nOf+1yWLVuWJPnGN76R9vb2fOc738nVV1897jMDAOPHE8AJ6Lvf/W4WLVqUD33oQ5k9e3YuuOCC3HnnnfXzL774YgYGBtLd3V1fa2try+LFi7Nt27Y3vOfw8HBqtdqoAwA4OQnACeinP/1ptmzZkrPPPjvf//7388lPfjKf/vSnc/fddydJBgYGkiTt7e2jXtfe3l4/95t6e3vT1tZWP+bNmze23wQAMGYE4AQ0MjKSCy+8MDfffHMuuOCCXHPNNfnEJz6RrVu3vuV7rlu3LkNDQ/Vj3759b+PEAMB4EoAT0Jw5c3LuueeOWjvnnHOyd+/eJElHR0eSZHBwcNQ1g4OD9XO/qaWlJa2traMOAODkJAAnoCVLlmTPnj2j1p5//vm8853vTPLqB0I6OjrS19dXP1+r1bJ9+/Z0dXWN66wAwPjzKeAJaPXq1bnkkkty88035y/+4i+yY8eO3HHHHbnjjjuSJE1NTVm1alW++MUv5uyzz05nZ2fWr1+fuXPn5sorr2zs8ADAmBOAE9BFF12UBx54IOvWrcvf//3fp7OzM7feemtWrFhRv+aGG27I4cOHc8011+TgwYO59NJL88gjj2TatGkNnBwAGA9NVVVVjR6Ck0+tVktbW1suy7I0N01p9DgA/JZeqY7l8TyYoaEh7+sukPcAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgBOQMePH8/69evT2dmZ6dOn513vele+8IUvpKqq+jVVVWXDhg2ZM2dOpk+fnu7u7rzwwgsNnBoAGC8CcALauHFjtmzZkttvvz3PPvtsNm7cmE2bNuW2226rX7Np06Zs3rw5W7duzfbt23PKKadk6dKlOXLkSAMnBwDGQ3OjB+Dt98Mf/jDLli3LFVdckSQ566yz8s1vfjM7duxI8urTv1tvvTWf+9znsmzZsiTJN77xjbS3t+c73/lOrr766obNDgCMPU8AJ6BLLrkkfX19ef7555MkP/7xj/PEE0/kgx/8YJLkxRdfzMDAQLq7u+uvaWtry+LFi7Nt27Y3vOfw8HBqtdqoAwA4OXkCOAGtXbs2tVotCxYsyOTJk3P8+PHcdNNNWbFiRZJkYGAgSdLe3j7qde3t7fVzv6m3tzef//znx3ZwAGBceAI4AX3729/OPffck3vvvTe7d+/O3XffnX/8x3/M3Xff/ZbvuW7dugwNDdWPffv2vY0TAwDjyRPACegzn/lM1q5dW38v3/nnn5//+q//Sm9vb1auXJmOjo4kyeDgYObMmVN/3eDgYN73vve94T1bWlrS0tIy5rMDAGPPE8AJ6OWXX86kSaP/006ePDkjIyNJks7OznR0dKSvr69+vlarZfv27enq6hrXWQGA8ecJ4AT0p3/6p7npppsyf/78vPvd785//Md/5Mtf/nI+9rGPJUmampqyatWqfPGLX8zZZ5+dzs7OrF+/PnPnzs2VV17Z2OEBgDEnACeg2267LevXr8+nPvWpHDhwIHPnzs1f//VfZ8OGDfVrbrjhhhw+fDjXXHNNDh48mEsvvTSPPPJIpk2b1sDJAYDx0FS99p+HgDepVqulra0tl2VZmpumNHocAH5Lr1TH8ngezNDQUFpbWxs9DuPMewABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAArT3OgBODlVVZUkeSXHkqrBwwDwW3slx5L8+s9zyiIAeUsOHTqUJHki/9bgSQD4XRw6dChtbW2NHoNx1lRJf96CkZGR7N+/P1VVZf78+dm3b19aW1sbPdYJrVarZd68efbqf2Gf3jx79ebZq9erqiqHDh3K3LlzM2mSd4SVxhNA3pJJkyblzDPPTK1WS5K0trb6Q/VNsldvjn168+zVm2evRvPkr1ySHwCgMAIQAKAwApDfSUtLS2688ca0tLQ0epQTnr16c+zTm2ev3jx7BaP5EAgAQGE8AQQAKIwABAAojAAEACiMAAQAKIwA5Hfy1a9+NWeddVamTZuWxYsXZ8eOHY0eqaF6e3tz0UUXZcaMGZk9e3auvPLK7NmzZ9Q1R44cSU9PT2bNmpVTTz01y5cvz+DgYIMmPjHccsstaWpqyqpVq+pr9unXXnrppXz4wx/OrFmzMn369Jx//vl56qmn6uerqsqGDRsyZ86cTJ8+Pd3d3XnhhRcaOHFjHD9+POvXr09nZ2emT5+ed73rXfnCF74w6t+6tVfwKgHIW/atb30ra9asyY033pjdu3fnve99b5YuXZoDBw40erSG6e/vT09PT5588sk8+uijOXbsWD7wgQ/k8OHD9WtWr16dhx56KPfff3/6+/uzf//+XHXVVQ2curF27tyZr3/963nPe94zat0+veqXv/xllixZkilTpuR73/tennnmmXzpS1/KaaedVr9m06ZN2bx5c7Zu3Zrt27fnlFNOydKlS3PkyJEGTj7+Nm7cmC1btuT222/Ps88+m40bN2bTpk257bbb6tfYK/gfFbxFF198cdXT01P/+vjx49XcuXOr3t7eBk51Yjlw4ECVpOrv76+qqqoOHjxYTZkypbr//vvr1zz77LNVkmrbtm2NGrNhDh06VJ199tnVo48+Wv3RH/1Rdf3111dVZZ9e67Of/Wx16aWX/j/Pj4yMVB0dHdU//MM/1NcOHjxYtbS0VN/85jfHY8QTxhVXXFF97GMfG7V21VVXVStWrKiqyl7Ba3kCyFty9OjR7Nq1K93d3fW1SZMmpbu7O9u2bWvgZCeWoaGhJMnpp5+eJNm1a1eOHTs2at8WLFiQ+fPnF7lvPT09ueKKK0btR2KfXuu73/1uFi1alA996EOZPXt2Lrjggtx555318y+++GIGBgZG7VVbW1sWL15c3F5dcskl6evry/PPP58k+fGPf5wnnngiH/zgB5PYK3it5kYPwMnpF7/4RY4fP5729vZR6+3t7XnuuecaNNWJZWRkJKtWrcqSJUty3nnnJUkGBgYyderUzJw5c9S17e3tGRgYaMCUjXPfffdl9+7d2blz5+vO2adf++lPf5otW7ZkzZo1+du//dvs3Lkzn/70pzN16tSsXLmyvh9v9HuxtL1au3ZtarVaFixYkMmTJ+f48eO56aabsmLFiiSxV/AaAhDGSE9PT55++uk88cQTjR7lhLNv375cf/31efTRRzNt2rRGj3NCGxkZyaJFi3LzzTcnSS644II8/fTT2bp1a1auXNng6U4s3/72t3PPPffk3nvvzbvf/e786Ec/yqpVqzJ37lx7Bb/BXwHzlpxxxhmZPHny6z6VOTg4mI6OjgZNdeK49tpr8/DDD+cHP/hBzjzzzPp6R0dHjh49moMHD466vrR927VrVw4cOJALL7wwzc3NaW5uTn9/fzZv3pzm5ua0t7fbp/8xZ86cnHvuuaPWzjnnnOzduzdJ6vvh92Lymc98JmvXrs3VV1+d888/P3/1V3+V1atXp7e3N4m9gtcSgLwlU6dOzcKFC9PX11dfGxkZSV9fX7q6uho4WWNVVZVrr702DzzwQB577LF0dnaOOr9w4cJMmTJl1L7t2bMne/fuLWrfLr/88vzkJz/Jj370o/qxaNGirFixov5r+/SqJUuWvO5HCT3//PN55zvfmSTp7OxMR0fHqL2q1WrZvn17cXv18ssvZ9Kk0f9bmzx5ckZGRpLYKxil0Z9C4eR13333VS0tLdU///M/V88880x1zTXXVDNnzqwGBgYaPVrDfPKTn6za2tqqxx9/vPrZz35WP15++eX6NX/zN39TzZ8/v3rssceqp556qurq6qq6uroaOPWJ4bWfAq4q+/R/7dixo2pubq5uuumm6oUXXqjuueee6h3veEf1L//yL/VrbrnllmrmzJnVgw8+WP3nf/5ntWzZsqqzs7P61a9+1cDJx9/KlSur3//9368efvjh6sUXX6z+9V//tTrjjDOqG264oX6NvYJXCUB+J7fddls1f/78aurUqdXFF19cPfnkk40eqaGSvOFx11131a/51a9+VX3qU5+qTjvttOod73hH9Wd/9mfVz372s8YNfYL4zQC0T7/20EMPVeedd17V0tJSLViwoLrjjjtGnR8ZGanWr19ftbe3Vy0tLdXll19e7dmzp0HTNk6tVquuv/76av78+dW0adOqP/iDP6j+7u/+rhoeHq5fY6/gVU1V9ZofkQ4AwITnPYAAAIURgAAAhRGAAACFEYAAAIURgAAAhRGAAACFEYAAAIURgAAAhRGAAACFEYAAAIURgAAAhRGAAACFEYAAAIURgAAAhRGAAACFEYAAAIURgAAAhRGAAACFEYAAAIURgAAAhRGAAACFEYAAAIURgAAAhRGAAACFEYAAAIURgAAAhRGAAACFEYAAAIURgAAAhRGAAACFEYAAAIX5PxXu2hUDUnl5AAAAAElFTkSuQmCC",
      "text/html": [
       "\n",
       "            <div style=\"display: inline-block;\">\n",
       "                <div class=\"jupyter-widgets widget-label\" style=\"text-align: center;\">\n",
       "                    Figure\n",
       "                </div>\n",
       "                <img src='data:image/png;base64,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SV9fX31tz5492bt3b7q6usZ7XABgnHkCOAH19PTk3nvvzYMPPpgZM2bU39fX1taW6dOnp62tLR//+MezZs2anH766Wltbc11112Xrq6uN/UJYADg5CYAJ6AtW7YkSS677LJR63fddVc+8pGPJEm+8pWvZNKkSVm+fHmGh4ezdOnSfO1rXxvnSQGARvBzAHlL/BxAgJObnwNYNu8BBAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwALMAtt9ySpqamrFq1qr525MiR9PT0ZNasWTn11FOzfPnyDA4ONm5IAGDcCMAJbufOnfn617+e97znPaPWV69enYceeij3339/+vv7s3///lx11VUNmhIAGE8CcAL77//+76xYsSJ33nlnTjvttPr60NBQ/umf/ilf/vKX88d//MdZuHBh7rrrrvzwhz/Mk08+2cCJAYDxIAAnsJ6enlxxxRXp7u4etb5r164cO3Zs1PqCBQsyf/78bNu2bbzHBADGWXOjB2Bs3Hfffdm9e3d27tz5unMDAwOZOnVqZs6cOWq9vb09AwMDb3i/4eHhDA8P17+u1Wpv67wAwPjxBHAC2rdvX66//vrcc889mTZt2ttyz97e3rS1tdWPefPmvS33BQDGnwCcgHbt2pUDBw7kwgsvTHNzc5qbm9Pf35/Nmzenubk57e3tOXr0aA4ePDjqdYODg+no6HjDe65bty5DQ0P1Y9++fePwnQAAY8FfAU9Al19+eX7yk5+MWvvoRz+aBQsW5LOf/WzmzZuXKVOmpK+vL8uXL0+S7NmzJ3v37k1XV9cb3rOlpSUtLS1jPjsAMPYE4AQ0Y8aMnHfeeaPWTjnllMyaNau+/vGPfzxr1qzJ6aefntbW1lx33XXp6urK+9///kaMDACMIwFYqK985SuZNGlSli9fnuHh4SxdujRf+9rXGj0WADAOmqqqqho9BCefWq2Wtra2XJZlaW6a0uhxAPgtvVIdy+N5MENDQ2ltbW30OIwzHwIBACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAJygXnrppXz4wx/OrFmzMn369Jx//vl56qmn6uerqsqGDRsyZ86cTJ8+Pd3d3XnhhRcaODEAMF4E4AT0y1/+MkuWLMmUKVPyve99L88880y+9KUv5bTTTqtfs2nTpmzevDlbt27N9u3bc8opp2Tp0qU5cuRIAycHAMZDc6MH4O23cePGzJs3L3fddVd9rbOzs/7rqqpy66235nOf+1yWLVuWJPnGN76R9vb2fOc738nVV1897jMDAOPHE8AJ6Lvf/W4WLVqUD33oQ5k9e3YuuOCC3HnnnfXzL774YgYGBtLd3V1fa2try+LFi7Nt27Y3vOfw8HBqtdqoAwA4OQnACeinP/1ptmzZkrPPPjvf//7388lPfjKf/vSnc/fddydJBgYGkiTt7e2jXtfe3l4/95t6e3vT1tZWP+bNmze23wQAMGYE4AQ0MjKSCy+8MDfffHMuuOCCXHPNNfnEJz6RrVu3vuV7rlu3LkNDQ/Vj3759b+PEAMB4EoAT0Jw5c3LuueeOWjvnnHOyd+/eJElHR0eSZHBwcNQ1g4OD9XO/qaWlJa2traMOAODkJAAnoCVLlmTPnj2j1p5//vm8853vTPLqB0I6OjrS19dXP1+r1bJ9+/Z0dXWN66wAwPjzKeAJaPXq1bnkkkty88035y/+4i+yY8eO3HHHHbnjjjuSJE1NTVm1alW++MUv5uyzz05nZ2fWr1+fuXPn5sorr2zs8ADAmBOAE9BFF12UBx54IOvWrcvf//3fp7OzM7feemtWrFhRv+aGG27I4cOHc8011+TgwYO59NJL88gjj2TatGkNnBwAGA9NVVVVjR6Ck0+tVktbW1suy7I0N01p9DgA/JZeqY7l8TyYoaEh7+sukPcAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgBOQMePH8/69evT2dmZ6dOn513vele+8IUvpKqq+jVVVWXDhg2ZM2dOpk+fnu7u7rzwwgsNnBoAGC8CcALauHFjtmzZkttvvz3PPvtsNm7cmE2bNuW2226rX7Np06Zs3rw5W7duzfbt23PKKadk6dKlOXLkSAMnBwDGQ3OjB+Dt98Mf/jDLli3LFVdckSQ566yz8s1vfjM7duxI8urTv1tvvTWf+9znsmzZsiTJN77xjbS3t+c73/lOrr766obNDgCMPU8AJ6BLLrkkfX19ef7555MkP/7xj/PEE0/kgx/8YJLkxRdfzMDAQLq7u+uvaWtry+LFi7Nt27Y3vOfw8HBqtdqoAwA4OXkCOAGtXbs2tVotCxYsyOTJk3P8+PHcdNNNWbFiRZJkYGAgSdLe3j7qde3t7fVzv6m3tzef//znx3ZwAGBceAI4AX3729/OPffck3vvvTe7d+/O3XffnX/8x3/M3Xff/ZbvuW7dugwNDdWPffv2vY0TAwDjyRPACegzn/lM1q5dW38v3/nnn5//+q//Sm9vb1auXJmOjo4kyeDgYObMmVN/3eDgYN73vve94T1bWlrS0tIy5rMDAGPPE8AJ6OWXX86kSaP/006ePDkjIyNJks7OznR0dKSvr69+vlarZfv27enq6hrXWQGA8ecJ4AT0p3/6p7npppsyf/78vPvd785//Md/5Mtf/nI+9rGPJUmampqyatWqfPGLX8zZZ5+dzs7OrF+/PnPnzs2VV17Z2OEBgDEnACeg2267LevXr8+nPvWpHDhwIHPnzs1f//VfZ8OGDfVrbrjhhhw+fDjXXHNNDh48mEsvvTSPPPJIpk2b1sDJAYDx0FS99p+HgDepVqulra0tl2VZmpumNHocAH5Lr1TH8ngezNDQUFpbWxs9DuPMewABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAArT3OgBODlVVZUkeSXHkqrBwwDwW3slx5L8+s9zyiIAeUsOHTqUJHki/9bgSQD4XRw6dChtbW2NHoNx1lRJf96CkZGR7N+/P1VVZf78+dm3b19aW1sbPdYJrVarZd68efbqf2Gf3jx79ebZq9erqiqHDh3K3LlzM2mSd4SVxhNA3pJJkyblzDPPTK1WS5K0trb6Q/VNsldvjn168+zVm2evRvPkr1ySHwCgMAIQAKAwApDfSUtLS2688ca0tLQ0epQTnr16c+zTm2ev3jx7BaP5EAgAQGE8AQQAKIwABAAojAAEACiMAAQAKIwA5Hfy1a9+NWeddVamTZuWxYsXZ8eOHY0eqaF6e3tz0UUXZcaMGZk9e3auvPLK7NmzZ9Q1R44cSU9PT2bNmpVTTz01y5cvz+DgYIMmPjHccsstaWpqyqpVq+pr9unXXnrppXz4wx/OrFmzMn369Jx//vl56qmn6uerqsqGDRsyZ86cTJ8+Pd3d3XnhhRcaOHFjHD9+POvXr09nZ2emT5+ed73rXfnCF74w6t+6tVfwKgHIW/atb30ra9asyY033pjdu3fnve99b5YuXZoDBw40erSG6e/vT09PT5588sk8+uijOXbsWD7wgQ/k8OHD9WtWr16dhx56KPfff3/6+/uzf//+XHXVVQ2curF27tyZr3/963nPe94zat0+veqXv/xllixZkilTpuR73/tennnmmXzpS1/KaaedVr9m06ZN2bx5c7Zu3Zrt27fnlFNOydKlS3PkyJEGTj7+Nm7cmC1btuT222/Ps88+m40bN2bTpk257bbb6tfYK/gfFbxFF198cdXT01P/+vjx49XcuXOr3t7eBk51Yjlw4ECVpOrv76+qqqoOHjxYTZkypbr//vvr1zz77LNVkmrbtm2NGrNhDh06VJ199tnVo48+Wv3RH/1Rdf3111dVZZ9e67Of/Wx16aWX/j/Pj4yMVB0dHdU//MM/1NcOHjxYtbS0VN/85jfHY8QTxhVXXFF97GMfG7V21VVXVStWrKiqyl7Ba3kCyFty9OjR7Nq1K93d3fW1SZMmpbu7O9u2bWvgZCeWoaGhJMnpp5+eJNm1a1eOHTs2at8WLFiQ+fPnF7lvPT09ueKKK0btR2KfXuu73/1uFi1alA996EOZPXt2Lrjggtx555318y+++GIGBgZG7VVbW1sWL15c3F5dcskl6evry/PPP58k+fGPf5wnnngiH/zgB5PYK3it5kYPwMnpF7/4RY4fP5729vZR6+3t7XnuuecaNNWJZWRkJKtWrcqSJUty3nnnJUkGBgYyderUzJw5c9S17e3tGRgYaMCUjXPfffdl9+7d2blz5+vO2adf++lPf5otW7ZkzZo1+du//dvs3Lkzn/70pzN16tSsXLmyvh9v9HuxtL1au3ZtarVaFixYkMmTJ+f48eO56aabsmLFiiSxV/AaAhDGSE9PT55++uk88cQTjR7lhLNv375cf/31efTRRzNt2rRGj3NCGxkZyaJFi3LzzTcnSS644II8/fTT2bp1a1auXNng6U4s3/72t3PPPffk3nvvzbvf/e786Ec/yqpVqzJ37lx7Bb/BXwHzlpxxxhmZPHny6z6VOTg4mI6OjgZNdeK49tpr8/DDD+cHP/hBzjzzzPp6R0dHjh49moMHD466vrR927VrVw4cOJALL7wwzc3NaW5uTn9/fzZv3pzm5ua0t7fbp/8xZ86cnHvuuaPWzjnnnOzduzdJ6vvh92Lymc98JmvXrs3VV1+d888/P3/1V3+V1atXp7e3N4m9gtcSgLwlU6dOzcKFC9PX11dfGxkZSV9fX7q6uho4WWNVVZVrr702DzzwQB577LF0dnaOOr9w4cJMmTJl1L7t2bMne/fuLWrfLr/88vzkJz/Jj370o/qxaNGirFixov5r+/SqJUuWvO5HCT3//PN55zvfmSTp7OxMR0fHqL2q1WrZvn17cXv18ssvZ9Kk0f9bmzx5ckZGRpLYKxil0Z9C4eR13333VS0tLdU///M/V88880x1zTXXVDNnzqwGBgYaPVrDfPKTn6za2tqqxx9/vPrZz35WP15++eX6NX/zN39TzZ8/v3rssceqp556qurq6qq6uroaOPWJ4bWfAq4q+/R/7dixo2pubq5uuumm6oUXXqjuueee6h3veEf1L//yL/VrbrnllmrmzJnVgw8+WP3nf/5ntWzZsqqzs7P61a9+1cDJx9/KlSur3//9368efvjh6sUXX6z+9V//tTrjjDOqG264oX6NvYJXCUB+J7fddls1f/78aurUqdXFF19cPfnkk40eqaGSvOFx11131a/51a9+VX3qU5+qTjvttOod73hH9Wd/9mfVz372s8YNfYL4zQC0T7/20EMPVeedd17V0tJSLViwoLrjjjtGnR8ZGanWr19ftbe3Vy0tLdXll19e7dmzp0HTNk6tVquuv/76av78+dW0adOqP/iDP6j+7u/+rhoeHq5fY6/gVU1V9ZofkQ4AwITnPYAAAIURgAAAhRGAAACFEYAAAIURgAAAhRGAAACFEYAAAIURgAAAhRGAAACFEYAAAIURgAAAhRGAAACFEYAAAIURgAAAhRGAAACFEYAAAIURgAAAhRGAAACFEYAAAIURgAAAhRGAAACFEYAAAIURgAAAhRGAAACFEYAAAIURgAAAhRGAAACFEYAAAIURgAAAhRGAAACFEYAAAIX5PxXu2hUDUnl5AAAAAElFTkSuQmCC' width=640.0/>\n",
       "            </div>\n",
       "        "
      ],
      "text/plain": [
       "Canvas(toolbar=Toolbar(toolitems=[('Home', 'Reset original view', 'home', 'home'), ('Back', 'Back to previous …"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig,ax,controls = ims.slider(kernel_array)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "98c7079b-a34a-4a4c-a298-1dbd5c83fad9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "0ba68aeeab9d4e11ac7bfeea137ad502",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "VBox(children=(HBox(children=(Play(value=0, max=98), IntSlider(value=0, description='axis0', max=98, readout=F…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "79a54f005c9640fa891ac453924ed05c",
       "version_major": 2,
       "version_minor": 0
      },
      "image/png": "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SV9fX31tz5492bt3b7q6usZ7XABgnHkCOAH19PTk3nvvzYMPPpgZM2bU39fX1taW6dOnp62tLR//+MezZs2anH766Wltbc11112Xrq6uN/UJYADg5CYAJ6AtW7YkSS677LJR63fddVc+8pGPJEm+8pWvZNKkSVm+fHmGh4ezdOnSfO1rXxvnSQGARvBzAHlL/BxAgJObnwNYNu8BBAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwALMAtt9ySpqamrFq1qr525MiR9PT0ZNasWTn11FOzfPnyDA4ONm5IAGDcCMAJbufOnfn617+e97znPaPWV69enYceeij3339/+vv7s3///lx11VUNmhIAGE8CcAL77//+76xYsSJ33nlnTjvttPr60NBQ/umf/ilf/vKX88d//MdZuHBh7rrrrvzwhz/Mk08+2cCJAYDxIAAnsJ6enlxxxRXp7u4etb5r164cO3Zs1PqCBQsyf/78bNu2bbzHBADGWXOjB2Bs3Hfffdm9e3d27tz5unMDAwOZOnVqZs6cOWq9vb09AwMDb3i/4eHhDA8P17+u1Wpv67wAwPjxBHAC2rdvX66//vrcc889mTZt2ttyz97e3rS1tdWPefPmvS33BQDGnwCcgHbt2pUDBw7kwgsvTHNzc5qbm9Pf35/Nmzenubk57e3tOXr0aA4ePDjqdYODg+no6HjDe65bty5DQ0P1Y9++fePwnQAAY8FfAU9Al19+eX7yk5+MWvvoRz+aBQsW5LOf/WzmzZuXKVOmpK+vL8uXL0+S7NmzJ3v37k1XV9cb3rOlpSUtLS1jPjsAMPYE4AQ0Y8aMnHfeeaPWTjnllMyaNau+/vGPfzxr1qzJ6aefntbW1lx33XXp6urK+9///kaMDACMIwFYqK985SuZNGlSli9fnuHh4SxdujRf+9rXGj0WADAOmqqqqho9BCefWq2Wtra2XJZlaW6a0uhxAPgtvVIdy+N5MENDQ2ltbW30OIwzHwIBACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAJygXnrppXz4wx/OrFmzMn369Jx//vl56qmn6uerqsqGDRsyZ86cTJ8+Pd3d3XnhhRcaODEAMF4E4AT0y1/+MkuWLMmUKVPyve99L88880y+9KUv5bTTTqtfs2nTpmzevDlbt27N9u3bc8opp2Tp0qU5cuRIAycHAMZDc6MH4O23cePGzJs3L3fddVd9rbOzs/7rqqpy66235nOf+1yWLVuWJPnGN76R9vb2fOc738nVV1897jMDAOPHE8AJ6Lvf/W4WLVqUD33oQ5k9e3YuuOCC3HnnnfXzL774YgYGBtLd3V1fa2try+LFi7Nt27Y3vOfw8HBqtdqoAwA4OQnACeinP/1ptmzZkrPPPjvf//7388lPfjKf/vSnc/fddydJBgYGkiTt7e2jXtfe3l4/95t6e3vT1tZWP+bNmze23wQAMGYE4AQ0MjKSCy+8MDfffHMuuOCCXHPNNfnEJz6RrVu3vuV7rlu3LkNDQ/Vj3759b+PEAMB4EoAT0Jw5c3LuueeOWjvnnHOyd+/eJElHR0eSZHBwcNQ1g4OD9XO/qaWlJa2traMOAODkJAAnoCVLlmTPnj2j1p5//vm8853vTPLqB0I6OjrS19dXP1+r1bJ9+/Z0dXWN66wAwPjzKeAJaPXq1bnkkkty88035y/+4i+yY8eO3HHHHbnjjjuSJE1NTVm1alW++MUv5uyzz05nZ2fWr1+fuXPn5sorr2zs8ADAmBOAE9BFF12UBx54IOvWrcvf//3fp7OzM7feemtWrFhRv+aGG27I4cOHc8011+TgwYO59NJL88gjj2TatGkNnBwAGA9NVVVVjR6Ck0+tVktbW1suy7I0N01p9DgA/JZeqY7l8TyYoaEh7+sukPcAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgBOQMePH8/69evT2dmZ6dOn513vele+8IUvpKqq+jVVVWXDhg2ZM2dOpk+fnu7u7rzwwgsNnBoAGC8CcALauHFjtmzZkttvvz3PPvtsNm7cmE2bNuW2226rX7Np06Zs3rw5W7duzfbt23PKKadk6dKlOXLkSAMnBwDGQ3OjB+Dt98Mf/jDLli3LFVdckSQ566yz8s1vfjM7duxI8urTv1tvvTWf+9znsmzZsiTJN77xjbS3t+c73/lOrr766obNDgCMPU8AJ6BLLrkkfX19ef7555MkP/7xj/PEE0/kgx/8YJLkxRdfzMDAQLq7u+uvaWtry+LFi7Nt27Y3vOfw8HBqtdqoAwA4OXkCOAGtXbs2tVotCxYsyOTJk3P8+PHcdNNNWbFiRZJkYGAgSdLe3j7qde3t7fVzv6m3tzef//znx3ZwAGBceAI4AX3729/OPffck3vvvTe7d+/O3XffnX/8x3/M3Xff/ZbvuW7dugwNDdWPffv2vY0TAwDjyRPACegzn/lM1q5dW38v3/nnn5//+q//Sm9vb1auXJmOjo4kyeDgYObMmVN/3eDgYN73vve94T1bWlrS0tIy5rMDAGPPE8AJ6OWXX86kSaP/006ePDkjIyNJks7OznR0dKSvr69+vlarZfv27enq6hrXWQGA8ecJ4AT0p3/6p7npppsyf/78vPvd785//Md/5Mtf/nI+9rGPJUmampqyatWqfPGLX8zZZ5+dzs7OrF+/PnPnzs2VV17Z2OEBgDEnACeg2267LevXr8+nPvWpHDhwIHPnzs1f//VfZ8OGDfVrbrjhhhw+fDjXXHNNDh48mEsvvTSPPPJIpk2b1sDJAYDx0FS99p+HgDepVqulra0tl2VZmpumNHocAH5Lr1TH8ngezNDQUFpbWxs9DuPMewABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAArT3OgBODlVVZUkeSXHkqrBwwDwW3slx5L8+s9zyiIAeUsOHTqUJHki/9bgSQD4XRw6dChtbW2NHoNx1lRJf96CkZGR7N+/P1VVZf78+dm3b19aW1sbPdYJrVarZd68efbqf2Gf3jx79ebZq9erqiqHDh3K3LlzM2mSd4SVxhNA3pJJkyblzDPPTK1WS5K0trb6Q/VNsldvjn168+zVm2evRvPkr1ySHwCgMAIQAKAwApDfSUtLS2688ca0tLQ0epQTnr16c+zTm2ev3jx7BaP5EAgAQGE8AQQAKIwABAAojAAEACiMAAQAKIwA5Hfy1a9+NWeddVamTZuWxYsXZ8eOHY0eqaF6e3tz0UUXZcaMGZk9e3auvPLK7NmzZ9Q1R44cSU9PT2bNmpVTTz01y5cvz+DgYIMmPjHccsstaWpqyqpVq+pr9unXXnrppXz4wx/OrFmzMn369Jx//vl56qmn6uerqsqGDRsyZ86cTJ8+Pd3d3XnhhRcaOHFjHD9+POvXr09nZ2emT5+ed73rXfnCF74w6t+6tVfwKgHIW/atb30ra9asyY033pjdu3fnve99b5YuXZoDBw40erSG6e/vT09PT5588sk8+uijOXbsWD7wgQ/k8OHD9WtWr16dhx56KPfff3/6+/uzf//+XHXVVQ2curF27tyZr3/963nPe94zat0+veqXv/xllixZkilTpuR73/tennnmmXzpS1/KaaedVr9m06ZN2bx5c7Zu3Zrt27fnlFNOydKlS3PkyJEGTj7+Nm7cmC1btuT222/Ps88+m40bN2bTpk257bbb6tfYK/gfFbxFF198cdXT01P/+vjx49XcuXOr3t7eBk51Yjlw4ECVpOrv76+qqqoOHjxYTZkypbr//vvr1zz77LNVkmrbtm2NGrNhDh06VJ199tnVo48+Wv3RH/1Rdf3111dVZZ9e67Of/Wx16aWX/j/Pj4yMVB0dHdU//MM/1NcOHjxYtbS0VN/85jfHY8QTxhVXXFF97GMfG7V21VVXVStWrKiqyl7Ba3kCyFty9OjR7Nq1K93d3fW1SZMmpbu7O9u2bWvgZCeWoaGhJMnpp5+eJNm1a1eOHTs2at8WLFiQ+fPnF7lvPT09ueKKK0btR2KfXuu73/1uFi1alA996EOZPXt2Lrjggtx555318y+++GIGBgZG7VVbW1sWL15c3F5dcskl6evry/PPP58k+fGPf5wnnngiH/zgB5PYK3it5kYPwMnpF7/4RY4fP5729vZR6+3t7XnuuecaNNWJZWRkJKtWrcqSJUty3nnnJUkGBgYyderUzJw5c9S17e3tGRgYaMCUjXPfffdl9+7d2blz5+vO2adf++lPf5otW7ZkzZo1+du//dvs3Lkzn/70pzN16tSsXLmyvh9v9HuxtL1au3ZtarVaFixYkMmTJ+f48eO56aabsmLFiiSxV/AaAhDGSE9PT55++uk88cQTjR7lhLNv375cf/31efTRRzNt2rRGj3NCGxkZyaJFi3LzzTcnSS644II8/fTT2bp1a1auXNng6U4s3/72t3PPPffk3nvvzbvf/e786Ec/yqpVqzJ37lx7Bb/BXwHzlpxxxhmZPHny6z6VOTg4mI6OjgZNdeK49tpr8/DDD+cHP/hBzjzzzPp6R0dHjh49moMHD466vrR927VrVw4cOJALL7wwzc3NaW5uTn9/fzZv3pzm5ua0t7fbp/8xZ86cnHvuuaPWzjnnnOzduzdJ6vvh92Lymc98JmvXrs3VV1+d888/P3/1V3+V1atXp7e3N4m9gtcSgLwlU6dOzcKFC9PX11dfGxkZSV9fX7q6uho4WWNVVZVrr702DzzwQB577LF0dnaOOr9w4cJMmTJl1L7t2bMne/fuLWrfLr/88vzkJz/Jj370o/qxaNGirFixov5r+/SqJUuWvO5HCT3//PN55zvfmSTp7OxMR0fHqL2q1WrZvn17cXv18ssvZ9Kk0f9bmzx5ckZGRpLYKxil0Z9C4eR13333VS0tLdU///M/V88880x1zTXXVDNnzqwGBgYaPVrDfPKTn6za2tqqxx9/vPrZz35WP15++eX6NX/zN39TzZ8/v3rssceqp556qurq6qq6uroaOPWJ4bWfAq4q+/R/7dixo2pubq5uuumm6oUXXqjuueee6h3veEf1L//yL/VrbrnllmrmzJnVgw8+WP3nf/5ntWzZsqqzs7P61a9+1cDJx9/KlSur3//9368efvjh6sUXX6z+9V//tTrjjDOqG264oX6NvYJXCUB+J7fddls1f/78aurUqdXFF19cPfnkk40eqaGSvOFx11131a/51a9+VX3qU5+qTjvttOod73hH9Wd/9mfVz372s8YNfYL4zQC0T7/20EMPVeedd17V0tJSLViwoLrjjjtGnR8ZGanWr19ftbe3Vy0tLdXll19e7dmzp0HTNk6tVquuv/76av78+dW0adOqP/iDP6j+7u/+rhoeHq5fY6/gVU1V9ZofkQ4AwITnPYAAAIURgAAAhRGAAACFEYAAAIURgAAAhRGAAACFEYAAAIURgAAAhRGAAACFEYAAAIURgAAAhRGAAACFEYAAAIURgAAAhRGAAACFEYAAAIURgAAAhRGAAACFEYAAAIURgAAAhRGAAACFEYAAAIURgAAAhRGAAACFEYAAAIURgAAAhRGAAACFEYAAAIURgAAAhRGAAACFEYAAAIX5PxXu2hUDUnl5AAAAAElFTkSuQmCC",
      "text/html": [
       "\n",
       "            <div style=\"display: inline-block;\">\n",
       "                <div class=\"jupyter-widgets widget-label\" style=\"text-align: center;\">\n",
       "                    Figure\n",
       "                </div>\n",
       "                <img src='data:image/png;base64,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SV9fX31tz5492bt3b7q6usZ7XABgnHkCOAH19PTk3nvvzYMPPpgZM2bU39fX1taW6dOnp62tLR//+MezZs2anH766Wltbc11112Xrq6uN/UJYADg5CYAJ6AtW7YkSS677LJR63fddVc+8pGPJEm+8pWvZNKkSVm+fHmGh4ezdOnSfO1rXxvnSQGARvBzAHlL/BxAgJObnwNYNu8BBAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwALMAtt9ySpqamrFq1qr525MiR9PT0ZNasWTn11FOzfPnyDA4ONm5IAGDcCMAJbufOnfn617+e97znPaPWV69enYceeij3339/+vv7s3///lx11VUNmhIAGE8CcAL77//+76xYsSJ33nlnTjvttPr60NBQ/umf/ilf/vKX88d//MdZuHBh7rrrrvzwhz/Mk08+2cCJAYDxIAAnsJ6enlxxxRXp7u4etb5r164cO3Zs1PqCBQsyf/78bNu2bbzHBADGWXOjB2Bs3Hfffdm9e3d27tz5unMDAwOZOnVqZs6cOWq9vb09AwMDb3i/4eHhDA8P17+u1Wpv67wAwPjxBHAC2rdvX66//vrcc889mTZt2ttyz97e3rS1tdWPefPmvS33BQDGnwCcgHbt2pUDBw7kwgsvTHNzc5qbm9Pf35/Nmzenubk57e3tOXr0aA4ePDjqdYODg+no6HjDe65bty5DQ0P1Y9++fePwnQAAY8FfAU9Al19+eX7yk5+MWvvoRz+aBQsW5LOf/WzmzZuXKVOmpK+vL8uXL0+S7NmzJ3v37k1XV9cb3rOlpSUtLS1jPjsAMPYE4AQ0Y8aMnHfeeaPWTjnllMyaNau+/vGPfzxr1qzJ6aefntbW1lx33XXp6urK+9///kaMDACMIwFYqK985SuZNGlSli9fnuHh4SxdujRf+9rXGj0WADAOmqqqqho9BCefWq2Wtra2XJZlaW6a0uhxAPgtvVIdy+N5MENDQ2ltbW30OIwzHwIBACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAJygXnrppXz4wx/OrFmzMn369Jx//vl56qmn6uerqsqGDRsyZ86cTJ8+Pd3d3XnhhRcaODEAMF4E4AT0y1/+MkuWLMmUKVPyve99L88880y+9KUv5bTTTqtfs2nTpmzevDlbt27N9u3bc8opp2Tp0qU5cuRIAycHAMZDc6MH4O23cePGzJs3L3fddVd9rbOzs/7rqqpy66235nOf+1yWLVuWJPnGN76R9vb2fOc738nVV1897jMDAOPHE8AJ6Lvf/W4WLVqUD33oQ5k9e3YuuOCC3HnnnfXzL774YgYGBtLd3V1fa2try+LFi7Nt27Y3vOfw8HBqtdqoAwA4OQnACeinP/1ptmzZkrPPPjvf//7388lPfjKf/vSnc/fddydJBgYGkiTt7e2jXtfe3l4/95t6e3vT1tZWP+bNmze23wQAMGYE4AQ0MjKSCy+8MDfffHMuuOCCXHPNNfnEJz6RrVu3vuV7rlu3LkNDQ/Vj3759b+PEAMB4EoAT0Jw5c3LuueeOWjvnnHOyd+/eJElHR0eSZHBwcNQ1g4OD9XO/qaWlJa2traMOAODkJAAnoCVLlmTPnj2j1p5//vm8853vTPLqB0I6OjrS19dXP1+r1bJ9+/Z0dXWN66wAwPjzKeAJaPXq1bnkkkty88035y/+4i+yY8eO3HHHHbnjjjuSJE1NTVm1alW++MUv5uyzz05nZ2fWr1+fuXPn5sorr2zs8ADAmBOAE9BFF12UBx54IOvWrcvf//3fp7OzM7feemtWrFhRv+aGG27I4cOHc8011+TgwYO59NJL88gjj2TatGkNnBwAGA9NVVVVjR6Ck0+tVktbW1suy7I0N01p9DgA/JZeqY7l8TyYoaEh7+sukPcAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgACABRGAAIAFEYAAgAURgBOQMePH8/69evT2dmZ6dOn513vele+8IUvpKqq+jVVVWXDhg2ZM2dOpk+fnu7u7rzwwgsNnBoAGC8CcALauHFjtmzZkttvvz3PPvtsNm7cmE2bNuW2226rX7Np06Zs3rw5W7duzfbt23PKKadk6dKlOXLkSAMnBwDGQ3OjB+Dt98Mf/jDLli3LFVdckSQ566yz8s1vfjM7duxI8urTv1tvvTWf+9znsmzZsiTJN77xjbS3t+c73/lOrr766obNDgCMPU8AJ6BLLrkkfX19ef7555MkP/7xj/PEE0/kgx/8YJLkxRdfzMDAQLq7u+uvaWtry+LFi7Nt27Y3vOfw8HBqtdqoAwA4OXkCOAGtXbs2tVotCxYsyOTJk3P8+PHcdNNNWbFiRZJkYGAgSdLe3j7qde3t7fVzv6m3tzef//znx3ZwAGBceAI4AX3729/OPffck3vvvTe7d+/O3XffnX/8x3/M3Xff/ZbvuW7dugwNDdWPffv2vY0TAwDjyRPACegzn/lM1q5dW38v3/nnn5//+q//Sm9vb1auXJmOjo4kyeDgYObMmVN/3eDgYN73vve94T1bWlrS0tIy5rMDAGPPE8AJ6OWXX86kSaP/006ePDkjIyNJks7OznR0dKSvr69+vlarZfv27enq6hrXWQGA8ecJ4AT0p3/6p7npppsyf/78vPvd785//Md/5Mtf/nI+9rGPJUmampqyatWqfPGLX8zZZ5+dzs7OrF+/PnPnzs2VV17Z2OEBgDEnACeg2267LevXr8+nPvWpHDhwIHPnzs1f//VfZ8OGDfVrbrjhhhw+fDjXXHNNDh48mEsvvTSPPPJIpk2b1sDJAYDx0FS99p+HgDepVqulra0tl2VZmpumNHocAH5Lr1TH8ngezNDQUFpbWxs9DuPMewABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAAojAAEACiMAAQAKIwABAArT3OgBODlVVZUkeSXHkqrBwwDwW3slx5L8+s9zyiIAeUsOHTqUJHki/9bgSQD4XRw6dChtbW2NHoNx1lRJf96CkZGR7N+/P1VVZf78+dm3b19aW1sbPdYJrVarZd68efbqf2Gf3jx79ebZq9erqiqHDh3K3LlzM2mSd4SVxhNA3pJJkyblzDPPTK1WS5K0trb6Q/VNsldvjn168+zVm2evRvPkr1ySHwCgMAIQAKAwApDfSUtLS2688ca0tLQ0epQTnr16c+zTm2ev3jx7BaP5EAgAQGE8AQQAKIwABAAojAAEACiMAAQAKIwA5Hfy1a9+NWeddVamTZuWxYsXZ8eOHY0eqaF6e3tz0UUXZcaMGZk9e3auvPLK7NmzZ9Q1R44cSU9PT2bNmpVTTz01y5cvz+DgYIMmPjHccsstaWpqyqpVq+pr9unXXnrppXz4wx/OrFmzMn369Jx//vl56qmn6uerqsqGDRsyZ86cTJ8+Pd3d3XnhhRcaOHFjHD9+POvXr09nZ2emT5+ed73rXfnCF74w6t+6tVfwKgHIW/atb30ra9asyY033pjdu3fnve99b5YuXZoDBw40erSG6e/vT09PT5588sk8+uijOXbsWD7wgQ/k8OHD9WtWr16dhx56KPfff3/6+/uzf//+XHXVVQ2curF27tyZr3/963nPe94zat0+veqXv/xllixZkilTpuR73/tennnmmXzpS1/KaaedVr9m06ZN2bx5c7Zu3Zrt27fnlFNOydKlS3PkyJEGTj7+Nm7cmC1btuT222/Ps88+m40bN2bTpk257bbb6tfYK/gfFbxFF198cdXT01P/+vjx49XcuXOr3t7eBk51Yjlw4ECVpOrv76+qqqoOHjxYTZkypbr//vvr1zz77LNVkmrbtm2NGrNhDh06VJ199tnVo48+Wv3RH/1Rdf3111dVZZ9e67Of/Wx16aWX/j/Pj4yMVB0dHdU//MM/1NcOHjxYtbS0VN/85jfHY8QTxhVXXFF97GMfG7V21VVXVStWrKiqyl7Ba3kCyFty9OjR7Nq1K93d3fW1SZMmpbu7O9u2bWvgZCeWoaGhJMnpp5+eJNm1a1eOHTs2at8WLFiQ+fPnF7lvPT09ueKKK0btR2KfXuu73/1uFi1alA996EOZPXt2Lrjggtx555318y+++GIGBgZG7VVbW1sWL15c3F5dcskl6evry/PPP58k+fGPf5wnnngiH/zgB5PYK3it5kYPwMnpF7/4RY4fP5729vZR6+3t7XnuuecaNNWJZWRkJKtWrcqSJUty3nnnJUkGBgYyderUzJw5c9S17e3tGRgYaMCUjXPfffdl9+7d2blz5+vO2adf++lPf5otW7ZkzZo1+du//dvs3Lkzn/70pzN16tSsXLmyvh9v9HuxtL1au3ZtarVaFixYkMmTJ+f48eO56aabsmLFiiSxV/AaAhDGSE9PT55++uk88cQTjR7lhLNv375cf/31efTRRzNt2rRGj3NCGxkZyaJFi3LzzTcnSS644II8/fTT2bp1a1auXNng6U4s3/72t3PPPffk3nvvzbvf/e786Ec/yqpVqzJ37lx7Bb/BXwHzlpxxxhmZPHny6z6VOTg4mI6OjgZNdeK49tpr8/DDD+cHP/hBzjzzzPp6R0dHjh49moMHD466vrR927VrVw4cOJALL7wwzc3NaW5uTn9/fzZv3pzm5ua0t7fbp/8xZ86cnHvuuaPWzjnnnOzduzdJ6vvh92Lymc98JmvXrs3VV1+d888/P3/1V3+V1atXp7e3N4m9gtcSgLwlU6dOzcKFC9PX11dfGxkZSV9fX7q6uho4WWNVVZVrr702DzzwQB577LF0dnaOOr9w4cJMmTJl1L7t2bMne/fuLWrfLr/88vzkJz/Jj370o/qxaNGirFixov5r+/SqJUuWvO5HCT3//PN55zvfmSTp7OxMR0fHqL2q1WrZvn17cXv18ssvZ9Kk0f9bmzx5ckZGRpLYKxil0Z9C4eR13333VS0tLdU///M/V88880x1zTXXVDNnzqwGBgYaPVrDfPKTn6za2tqqxx9/vPrZz35WP15++eX6NX/zN39TzZ8/v3rssceqp556qurq6qq6uroaOPWJ4bWfAq4q+/R/7dixo2pubq5uuumm6oUXXqjuueee6h3veEf1L//yL/VrbrnllmrmzJnVgw8+WP3nf/5ntWzZsqqzs7P61a9+1cDJx9/KlSur3//9368efvjh6sUXX6z+9V//tTrjjDOqG264oX6NvYJXCUB+J7fddls1f/78aurUqdXFF19cPfnkk40eqaGSvOFx11131a/51a9+VX3qU5+qTjvttOod73hH9Wd/9mfVz372s8YNfYL4zQC0T7/20EMPVeedd17V0tJSLViwoLrjjjtGnR8ZGanWr19ftbe3Vy0tLdXll19e7dmzp0HTNk6tVquuv/76av78+dW0adOqP/iDP6j+7u/+rhoeHq5fY6/gVU1V9ZofkQ4AwITnPYAAAIURgAAAhRGAAACFEYAAAIURgAAAhRGAAACFEYAAAIURgAAAhRGAAACFEYAAAIURgAAAhRGAAACFEYAAAIURgAAAhRGAAACFEYAAAIURgAAAhRGAAACFEYAAAIURgAAAhRGAAACFEYAAAIURgAAAhRGAAACFEYAAAIURgAAAhRGAAACFEYAAAIURgAAAhRGAAACFEYAAAIX5PxXu2hUDUnl5AAAAAElFTkSuQmCC' width=640.0/>\n",
       "            </div>\n",
       "        "
      ],
      "text/plain": [
       "Canvas(toolbar=Toolbar(toolitems=[('Home', 'Reset original view', 'home', 'home'), ('Back', 'Back to previous …"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig,ax,controls = ims.slider(kernel_array)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "7d75a888-2b22-44d2-befe-4de65c9431ba",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(49, 49, 49)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "(int(np.floor(len(kernel_array)/2)),int(np.floor(len(kernel_array[0])/2)),int(np.floor(len(kernel_array[0][0])/2)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "471b7a56-bbb3-4973-9f71-2002d7950313",
   "metadata": {},
   "outputs": [],
   "source": [
    "my_dose_array = pickle.load(open('dose_1.pickle','rb'))\n",
    "mc_dose_array_full = BinnedResult('../Topas/TestingDose1.csv')\n",
    "mc_dose_array = mc_dose_array_full.data['Sum']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "4f211284-1084-4ee0-9412-a58f38a25b60",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "540014a4c4fe4cb59f97ab440723f5a7",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "VBox(children=(HBox(children=(Play(value=0, max=8), IntSlider(value=0, description='axis0', max=8, readout=Fal…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "01d9f5128c7c4dbea51f1e722b5e61f4",
       "version_major": 2,
       "version_minor": 0
      },
      "image/png": "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IbZs2dJ9jBs3LqZPnx5btmwRfx/BpEmTDvhWRNu2bYtjjz22oEW16f33348+ffb/q7q+vj66uroKWsSnhTuAFG7OnDkxY8aMGDduXJx66qmxePHi2LNnT8ycObPoaTWjpaUlli9fHg8//HA0NjbGjh07IiKiqakp+vbtW/C62tHY2HjA5yYPP/zw6N+/v89TfkRXX311nHbaabFgwYL4+te/HuvXr4877rgj7rjjjqKn1ZTzzjsvbrzxxhg6dGiMGjUqnnvuubj55ptj1qxZRU+jxvk2MPxHWLJkSdx0002xY8eOOOmkk+IXv/hFTJgwoehZNaOuru6g5+++++645JJLqjvmU2by5Mm+DczH9Oijj8a8efPi5ZdfjuHDh8ecOXPisssuK3pWTdm9e3fMnz8/Vq5cGbt27YrBgwfHRRddFNdcc00ceuihRc+jhglAAIBkfAYQACAZAQgAkIwABABIRgACACQjAAEAkhGAAADJCEAAgGQEIABAMgIQACAZAQgAkIwABABIRgACACQjAAEAkhGAAADJCEAAgGQEIABAMgIQACAZAQgAkIwABABIRgACACQjAAEAkhGAAADJCEAAgGQEIABAMgIQACAZAQgAkIwABABIRgACACQjAAEAkhGAAADJCEAAgGT+L5Zmta7K2i7+AAAAAElFTkSuQmCC",
      "text/html": [
       "\n",
       "            <div style=\"display: inline-block;\">\n",
       "                <div class=\"jupyter-widgets widget-label\" style=\"text-align: center;\">\n",
       "                    Figure\n",
       "                </div>\n",
       "                <img src='data:image/png;base64,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IbZs2dJ9jBs3LqZPnx5btmwRfx/BpEmTDvhWRNu2bYtjjz22oEW16f33348+ffb/q7q+vj66uroKWsSnhTuAFG7OnDkxY8aMGDduXJx66qmxePHi2LNnT8ycObPoaTWjpaUlli9fHg8//HA0NjbGjh07IiKiqakp+vbtW/C62tHY2HjA5yYPP/zw6N+/v89TfkRXX311nHbaabFgwYL4+te/HuvXr4877rgj7rjjjqKn1ZTzzjsvbrzxxhg6dGiMGjUqnnvuubj55ptj1qxZRU+jxvk2MPxHWLJkSdx0002xY8eOOOmkk+IXv/hFTJgwoehZNaOuru6g5+++++645JJLqjvmU2by5Mm+DczH9Oijj8a8efPi5ZdfjuHDh8ecOXPisssuK3pWTdm9e3fMnz8/Vq5cGbt27YrBgwfHRRddFNdcc00ceuihRc+jhglAAIBkfAYQACAZAQgAkIwABABIRgACACQjAAEAkhGAAADJCEAAgGQEIABAMgIQACAZAQgAkIwABABIRgACACQjAAEAkhGAAADJCEAAgGQEIABAMgIQACAZAQgAkIwABABIRgACACQjAAEAkhGAAADJCEAAgGQEIABAMgIQACAZAQgAkIwABABIRgACACQjAAEAkhGAAADJCEAAgGT+L5Zmta7K2i7+AAAAAElFTkSuQmCC' width=640.0/>\n",
       "            </div>\n",
       "        "
      ],
      "text/plain": [
       "Canvas(toolbar=Toolbar(toolitems=[('Home', 'Reset original view', 'home', 'home'), ('Back', 'Back to previous …"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig,ax,controls = ims.slider(dose)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "cdb0af15-94e6-40c2-8161-32429903e1b2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x7fc7e994a438>"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d7d81963ecfc4f89b40eb8deeae53bce",
       "version_major": 2,
       "version_minor": 0
      },
      "image/png": "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CEAAgGQEIABAMgIQACAZAQgAkIwABABIRgACACQjAAEAkhGAAADJCEAAgGQEIABAMgIQACAZAQgAkIwATOqII46ISqWy09HX11d6GgDQZF2lB1DG2rVrY3h4eOTxxo0b453vfGece+65BVcBAK0gAJM6+OCDd3h8xRVXxJFHHhlvfetbCy0CAFrFS8DEtm3b4pZbbokLLrggKpVK6TkAQJO5A0jceeed8fe//z3OP//8f3tNrVaLWq028nhoaKgFywCAZnAHkLjhhhti9uzZMWnSpH97TX9/f/T09Iwcvb29LVwIAIwmAZjc73//+3jggQfiwx/+8Itet2jRohgcHBw5BgYGWrQQABhtXgJO7sYbb4wJEybEGWec8aLXVavVqFarLVoFADSTO4CJ1ev1uPHGG2Pu3LnR1eV7AQDIQgAm9sADD8QzzzwTF1xwQekpAEALue2T2Omnnx6NRqP0DACgxdwBBABIRgACACQjAAEAkhGAAADJCEAAgGQEIABAMgIQACAZAQgAkIwABABIRgACACQjAAEAkhGAAADJCEAAgGQEIABAMl2lB9Dm6sMRlb34+4hKpfSC3WoMl14ANMVe//mnEtEovYFS9uKv3AAANIMABABIRgACACQjAAEAkhGAAADJCEAAgGQEIABAMgIQACAZAQgAkIwABABIRgACACQjAAEAkhGAAADJCEAAgGQEIABAMgIwqeHh4bj44otjypQpMXbs2DjyyCPjsssui0ajUXoaANBkXaUHUMaVV14Zy5Yti5tvvjmmTp0a69ati3nz5kVPT09cdNFFpecBAE0kAJN6+OGH4+yzz44zzjgjIiKOOOKIuO222+LRRx8tvAwAaDYvASd16qmnxsqVK+PJJ5+MiIif/exn8dBDD8Xs2bMLLwMAms0dwKQWLlwYQ0NDcdRRR0VnZ2cMDw/H4sWLY86cObu8vlarRa1WG3k8NDTUqqkAwChzBzCp7373u3HrrbfG8uXLY8OGDXHzzTfHV7/61bj55pt3eX1/f3/09PSMHL29vS1eDACMlkrDX/tMqbe3NxYuXBh9fX0j5770pS/FLbfcEr/61a92un5XdwB7e3vjtDg7uir7tGTzHqlUSi8A2Cu90NgeDzbujMHBwRg3blzpObSYl4CT+sc//hEdHTveAO7s7Ix6vb7L66vValSr1VZMAwCaTAAmddZZ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",
      "text/html": [
       "\n",
       "            <div style=\"display: inline-block;\">\n",
       "                <div class=\"jupyter-widgets widget-label\" style=\"text-align: center;\">\n",
       "                    Figure\n",
       "                </div>\n",
       "                <img src='data:image/png;base64,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CEAAgGQEIABAMgIQACAZAQgAkIwABABIRgACACQjAAEAkhGAAADJCEAAgGQEIABAMgIQACAZAQgAkIwATOqII46ISqWy09HX11d6GgDQZF2lB1DG2rVrY3h4eOTxxo0b453vfGece+65BVcBAK0gAJM6+OCDd3h8xRVXxJFHHhlvfetbCy0CAFrFS8DEtm3b4pZbbokLLrggKpVK6TkAQJO5A0jceeed8fe//z3OP//8f3tNrVaLWq028nhoaKgFywCAZnAHkLjhhhti9uzZMWnSpH97TX9/f/T09Iwcvb29LVwIAIwmAZjc73//+3jggQfiwx/+8Itet2jRohgcHBw5BgYGWrQQABhtXgJO7sYbb4wJEybEGWec8aLXVavVqFarLVoFADSTO4CJ1ev1uPHGG2Pu3LnR1eV7AQDIQgAm9sADD8QzzzwTF1xwQekpAEALue2T2Omnnx6NRqP0DACgxdwBBABIRgACACQjAAEAkhGAAADJCEAAgGQEIABAMgIQACAZAQgAkIwABABIRgACACQjAAEAkhGAAADJCEAAgGQEIABAMl2lB9Dm6sMRlb34+4hKpfSC3WoMl14ANMVe//mnEtEovYFS9uKv3AAANIMABABIRgACACQjAAEAkhGAAADJCEAAgGQEIABAMgIQACAZAQgAkIwABABIRgACACQjAAEAkhGAAADJCEAAgGQEIABAMgIwqeHh4bj44otjypQpMXbs2DjyyCPjsssui0ajUXoaANBkXaUHUMaVV14Zy5Yti5tvvjmmTp0a69ati3nz5kVPT09cdNFFpecBAE0kAJN6+OGH4+yzz44zzjgjIiKOOOKIuO222+LRRx8tvAwAaDYvASd16qmnxsqVK+PJJ5+MiIif/exn8dBDD8Xs2bMLLwMAms0dwKQWLlwYQ0NDcdRRR0VnZ2cMDw/H4sWLY86cObu8vlarRa1WG3k8NDTUqqkAwChzBzCp7373u3HrrbfG8uXLY8OGDXHzzTfHV7/61bj55pt3eX1/f3/09PSMHL29vS1eDACMlkrDX/tMqbe3NxYuXBh9fX0j5770pS/FLbfcEr/61a92un5XdwB7e3vjtDg7uir7tGTzHqlUSi8A2Cu90NgeDzbujMHBwRg3blzpObSYl4CT+sc//hEdHTveAO7s7Ix6vb7L66vValSr1VZMAwCaTAAmddZZ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' width=640.0/>\n",
       "            </div>\n",
       "        "
      ],
      "text/plain": [
       "Canvas(toolbar=Toolbar(toolitems=[('Home', 'Reset original view', 'home', 'home'), ('Back', 'Back to previous …"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "close(5);figure(5)\n",
    "imshow(dose[4])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "f09adf6e-ac10-46a6-9ab4-b139228321a8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.03631941168294036 0.0009266880412018293\n"
     ]
    }
   ],
   "source": [
    "print(sum(my_dose_array)*1.602e-10,sum(mc_dose_array))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "74cee703-c011-4813-a3f2-7ebb7bc03f51",
   "metadata": {
    "tags": []
   },
   "source": [
    "### Tests for Superposition"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "8ed17873-a185-4428-853b-f968e8678d96",
   "metadata": {},
   "outputs": [],
   "source": [
    "test_kernel = array([[[0,0,0],[0,0.1,0],[0,0,0]],\n",
    "                     [[0,0.1,0],[0.1,0.4,0.1],[0,0.1,0]],\n",
    "                     [[0,0,0],[0,0.1,0],[0,0,0]]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "e4769d3d-bd53-4826-bae9-33cc7feadc3e",
   "metadata": {},
   "outputs": [],
   "source": [
    "test_array_1 = array([[[1,1],[1,1]],\n",
    "                      [[1,1],[1,1]]])\n",
    "\n",
    "test_array_2 = array([[[1,1],[0,0]],\n",
    "                      [[0,0],[0,0]]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "4009f8da-ebf2-40e2-a981-6e1b6ee80c4c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[{'indices': (1, 1, 1), 'TERMA': 1, 'd': 0.1},\n",
       " {'indices': (1, 1, 2), 'TERMA': 1, 'd': 0.1},\n",
       " {'indices': (1, 2, 1), 'TERMA': 0, 'd': 0.1},\n",
       " {'indices': (1, 2, 2), 'TERMA': 0, 'd': 0.1},\n",
       " {'indices': (2, 1, 1), 'TERMA': 0, 'd': 0.1},\n",
       " {'indices': (2, 1, 2), 'TERMA': 0, 'd': 0.1},\n",
       " {'indices': (2, 2, 1), 'TERMA': 0, 'd': 0.1},\n",
       " {'indices': (2, 2, 2), 'TERMA': 0, 'd': 0.1}]"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_array = test_array_2\n",
    "\n",
    "voxel_info_1 = []\n",
    "n=0\n",
    "for x in range(len(test_array)):\n",
    "    for y in range(len(test_array[0])):\n",
    "        for z in range(len(test_array[0][0])):\n",
    "            voxel_info_1.append({})\n",
    "            voxel_info_1[n]['indices'] = (x+1,y+1,z+1)\n",
    "            voxel_info_1[n]['TERMA'] = test_array[x][y][z]\n",
    "            voxel_info_1[n]['d'] = 0.1\n",
    "            n += 1\n",
    "\n",
    "voxel_info_1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "02136eae-f9be-4f66-98e1-a5c8c934e689",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[[0.50758621, 0.50758621],\n",
       "        [0.20965517, 0.20965517]],\n",
       "\n",
       "       [[0.20965517, 0.20965517],\n",
       "        [0.07310345, 0.07310345]]])"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sd.Superposition(test_kernel,(3,3,3),(3,3,3),(0.75,0.75,0.75),[voxel_info_1],4)\n",
    "# sd.Superposition?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "c13e11a3-f7ca-436b-ba2c-6d449f84f1a3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[[0.50758621, 0.50758621],\n",
       "        [0.20965517, 0.20965517]],\n",
       "\n",
       "       [[0.20965517, 0.20965517],\n",
       "        [0.07310345, 0.07310345]]])"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x1,x2 = (2,2.1)\n",
    "y1,y2 = (0,0)\n",
    "z1,z2 = (1,1)\n",
    "\n",
    "dc.Superposition(test_kernel,(3,3,3),(3,3,3),(0.75,0.75,0.75),[voxel_info_1],[((x1,x2),(y1,y2),(z1,z2))],(8,8,8),4)\n",
    "# dc.Superposition?"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

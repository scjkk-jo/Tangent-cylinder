#!/usr/bin/env python
# coding: utf-8

# ### Processes a series of timesteps of a spherical shell simulation
# 
# Returns:    
# 
# 1) Kinetic energy, Reynolds number, Nusselt number (global) (.csv file)
# 
# 2) Coordinates, velocity components, temperature and their gradients in cylindrical basis (parquet files), within the tangent cylinder

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import mpl_toolkits.mplot3d.axes3d as axes3d
import pandas as pd
from os.path import normpath
from scipy.integrate import quad, dblquad, tplquad
#from scipy.integrate import nquad
from scipy.interpolate import RegularGridInterpolator
from scipy import integrate
import time
import netCDF4 as nc
import csv
import os
import glob
from datetime import datetime
from scipy.integrate import simps
import pyarrow as pa
import pyarrow.parquet as pq
import subprocess 
from scipy.interpolate import interp2d
from scipy.special import sph_harm


# In[2]:


def basic_settings():
    # path to folder containing input files
    input_path = "C:/path/to/input/"
    
    # path to output folder
    output_path = "C:/path/to/output/"
    
    # number of points in each dimension for cylindrical array
    num = 100

    return input_path, output_path, num

input_path, output_path, num = basic_settings()


# In[3]:


def output_grid(r, num):
    # output array: cylindrical. Choose TC or full width of spherical shell
    #s = np.arange(r.min()/num, r.min() + (r.min()/num), r.min()/num)  #TC
    s = np.arange(r.min()/num, r.max() + (r.min()/num), r.max()/num) # full width of shell
    phi = np.arange(0, 2*np.pi, (2*np.pi)/num)
    z = np.arange(0, r.max(), r.max()/num)  # nearly full height, allowing interpolation close to boundary
    
    # output array: cube. Not set up.
    #x = 
    #y = 
    #z = 
    
    return s, phi, z  #cylinder
    #return x, y, z  #cube


# In[4]:


#create a link to input files in working folder

def create_links(input_path):
    input_list_paths = []
    input_list_source = [] 
    remote_files = glob.glob(input_path + "*state*")  #insert essential part of input filename
    
    for file in remote_files:
        input_list_paths.append(subprocess.run( ['scp', '-qC', file, '.']))
        input_list_source.append(file)
    
    return input_list_source

input_list_source = create_links(input_path)
#print(input_list_source)


# In[5]:


def create_input_list(input_path):
    input_list_paths = []
    input_list = [] 
    for timestep in glob.glob(input_path + '*state*'):  #insert essential part of input filename
        input_list_paths.append(timestep)   
    for i in range(len(input_list_paths)):
        input_list.append(os.path.basename(input_list_paths[i]))
    input_list.sort()
    return input_list

input_list = create_input_list(input_path)
#print(input_list)


# In[6]:


def create_variables(coreData):
    r = np.ma.getdata(coreData.variables['r'][:])
    az = np.ma.getdata(coreData.variables['ph'][:])
    el = np.ma.getdata(coreData.variables['th'][:])
    temp = np.ma.getdata(coreData.variables['C'][:])
    u_r = np.ma.getdata(coreData.variables['ur'][:])
    u_el = np.ma.getdata(coreData.variables['uth'][:])
    u_az = np.ma.getdata(coreData.variables['uph'][:])
    temp_grad_el = np.ma.getdata(coreData.variables['Ct'][:])
    temp_grad_r = np.ma.getdata(coreData.variables['Cr'][:])
    
    #reverse arrays to convert latitude to polar coordinate
    el_new = el[::-1]
    u_r_new = u_r[:, :, ::-1]
    u_az_new = u_az[:, :, ::-1]
    u_el_new = u_el[:, :, ::-1]
    temp_new = temp[:, :, ::-1]
    temp_grad_el_new = temp_grad_el[:, :, ::-1]
    temp_grad_r_new = temp_grad_r[:, :, ::-1]
    
    # to be used for magnetic simulations
    icr = np.ma.getdata(coreData.variables['icr'][:])
    Br = np.ma.getdata(coreData.variables['Br'][:])
    B_el = np.ma.getdata(coreData.variables['Bth'][:])
    B_az = np.ma.getdata(coreData.variables['Bph'][:])
    icBT = np.ma.getdata(coreData.variables['icBT'][:])
    icBP = np.ma.getdata(coreData.variables['icBP'][:])
    
    return el_new, r, az, temp_new, u_r_new, u_el_new, u_az_new, temp_grad_el_new, temp_grad_r_new    


# In[7]:


# create grid with radial, azimuthal and elevation coordinates
def create_grid(r, az, el):
    r1 = np.zeros(len(r) * len(az) * len(el))
    az1 = np.zeros(len(r) * len(az) * len(el))
    el1 = np.zeros(len(r) * len(az) * len(el))
    l = 0
    for i in range(len(r)):
        for j in range(len(az)):
            for k in range(len(el)):
                r1[l] = r[i]
                az1[l] = az[j]
                el1[l] = el[k]
                l = l + 1  
    
    cmb_mask = r1 > r.max() - 0.00001 # select cmb
    icb_mask = r1 < r.min() + 0.00001 # select icb

    return r1, az1, el1, cmb_mask, icb_mask


# In[8]:


def interpolation_even_grid(el, r, az, temp, u_r, u_el, u_az):
    # interpolation functions
    interp_temp = RegularGridInterpolator((r, az, el), temp, method="linear", bounds_error=False)
    interp_ur = RegularGridInterpolator((r, az, el), u_r, method="linear", bounds_error=False)
    interp_uel = RegularGridInterpolator((r, az, el), u_el, method="linear", bounds_error=False)
    interp_uaz = RegularGridInterpolator((r, az, el), u_az, method="linear", bounds_error=False)
    
    # even grid: full sphere no points at poles
    r_check = np.linspace(r.min(), r.max(), len(r))
    el_check = np.linspace( 0 + (np.pi)/len(el), np.pi - (np.pi)/len(el), (len(el)))
    az_check = az # this is evenly spaced anyway for these simulations
    
    # make an even grid
    rr = np.zeros(len(r_check) * len(az_check) * len(el_check))
    pp = np.zeros(len(r_check) * len(az_check) * len(el_check))
    tt = np.zeros(len(r_check) * len(az_check) * len(el_check))
    
    l = 0
    for i in range(len(r_check)):
        for j in range(len(az_check)):
            for k in range(len(el_check)):
                rr[l] = r_check[i]
                pp[l] = az_check[j]
                tt[l] = el_check[k]
                l = l + 1
                
    boundary_mask = rr > rr.max() - 0.001 # to select values inside sphere
    
    # interpolate values onto even grid
    i_temp = interp_temp((rr, pp, tt))
    i_ur = interp_ur((rr, pp, tt))
    i_uel = interp_uel((rr, pp, tt))
    i_uaz = interp_uaz((rr, pp, tt))
        
    return i_temp, i_ur, i_uel, i_uaz, rr, pp, tt, r_check, az_check, el_check 


# In[9]:


# integrand for KE (trapz, simpson)
def KE1(u_r, u_az, u_el, r, el):
    return r**2 * np.sin(el) * (u_r**2 + u_az**2 + u_el**2)


# ### Find global kinetic energy and Reynolds number
# 
# $E_k = \frac{1}{2} \int_V |u|^2\ dV$
# 
# $Re =  \sqrt{\frac{2 E_k}{V}}$

# In[10]:


def KE_Re(r1, az1, el1, u_r, u_el, u_az, r, az, el):
    
    # gridded coordinate arrays, reshaped
    r2 = r1.reshape(len(r), len(az), len(el))
    el2 = el1.reshape(len(r), len(az), len(el))
    
    integrand1 = KE1(u_r, u_az, u_el, r2, el2)
    
    #trapz
    integral1 = np.trapz(np.trapz(np.trapz(integrand1, el, axis=2), az, axis=1), r, axis=0)
    
    #simpson
    integral2 = simps (  [simps  (  [simps( value, el) for value in integrand1], az ) ], r)
    
    K_E1 = 0.5 * integral1
    K_E2 = 0.5 * integral2
    
    shell_volume = (4/3 * np.pi * r.max()**3) - (4/3 * np.pi * r.min()**3)
    
    Re1 = np.sqrt(( 2 * K_E1)/ shell_volume)
    Re2 = np.sqrt(( 2 * K_E2)/ shell_volume)
    
    return K_E1, Re1, K_E2[0], Re2[0] 


# In[11]:


def cyl2sph(s, phi, z):
    r = np.sqrt(s**2 + z**2)
    az = phi
    el = np.arccos(z/np.sqrt(s**2 + z**2))
    return r, az, el


# In[12]:


def convert_vec_sph2cyl(r, az, el,  u_r, u_az, u_el):
    ux = u_r * np.sin(el) * np.cos(az) + u_az * (- np.sin(az)) + u_el * np.cos(el) * np.cos(az)
    uy = u_r * np.sin(el) * np.sin(az) + u_az *  np.cos(az) + u_el * np.cos(el) * np.sin(az)
    uz = u_r * np.cos(el) + u_el * - np.sin(el)
    u_s = ux * np.cos(az) + uy * np.sin(az)
    u_phi = (- ux) * np.sin(az) + uy * np.cos(az)
    u_z = uz
    return u_s, u_phi, u_z


# In[13]:


def interpolation_TC_grid(num, r, az, el, temp, u_r, u_el, u_az, s, phi, z):
    
    # make cylindrical grid
    ss_tc = np.zeros(len(s) * len(phi) * len(z))
    pp_tc = np.zeros(len(s) * len(phi) * len(z))
    zz_tc = np.zeros(len(s) * len(phi) * len(z))
    
    l = 0
    for i in range(len(s)):
        for j in range(len(phi)):
            for k in range(len(z)):
                ss_tc[l] = s[i]
                pp_tc[l] = phi[j]
                zz_tc[l] = z[k]
                l = l + 1
    TC_curved_surface = ss_tc > ss_tc.max() - 0.001 # Select values inside TC
    
    # convert to spherical form
    rr_tc, aa_tc, tt_tc = cyl2sph(ss_tc, pp_tc, zz_tc)
    
    TC_CMB = rr_tc > r.max() - 0.001 # select values closest to cmb
    TC_ICB = rr_tc < r.min() + 0.01  # select values closest to icb
    
    interp_temp = RegularGridInterpolator((r, az, el), temp, method="linear", bounds_error=False)
    interp_ur = RegularGridInterpolator((r, az, el), u_r, method="linear", bounds_error=False)
    interp_uel = RegularGridInterpolator((r, az, el), u_el, method="linear", bounds_error=False)
    interp_uaz = RegularGridInterpolator((r, az, el), u_az, method="linear", bounds_error=False)
    
    # interpolate values onto TC grid
    i_temp_tc = interp_temp((rr_tc, aa_tc, tt_tc))
    i_ur_tc = interp_ur((rr_tc, aa_tc, tt_tc))
    i_uel_tc = interp_uel((rr_tc, aa_tc, tt_tc))
    i_uaz_tc = interp_uaz((rr_tc, aa_tc, tt_tc))
    
    
    # convert the temp and velocities to cylindrical form
    u_s, u_phi, u_z = convert_vec_sph2cyl(rr_tc, aa_tc, tt_tc, i_ur_tc, i_uaz_tc, i_uel_tc)
    
    return ss_tc, pp_tc, zz_tc, u_s, u_phi, u_z, rr_tc, tt_tc, i_ur_tc, i_uaz_tc,            i_uel_tc, i_temp_tc, TC_curved_surface, TC_CMB, TC_ICB


# In[14]:


def gradients(s, phi, z, i_temp_tc, u_s, u_phi, u_z  ):
    i_temp = i_temp_tc.reshape(len(s), len(phi), len(z))
    i_us = u_s.reshape(len(s), len(phi), len(z))
    i_uphi = u_phi.reshape(len(s), len(phi), len(z))
    i_uz = u_z.reshape(len(s), len(phi), len(z))
    
    # spacing: not necessary for this grid but if I changed the spacing...
    s_space = s[1] - s[0]
    phi_space = phi[1] - phi[0]
    z_space = z[1] - z[0]
    
    tempgrad_s1 = np.gradient(i_temp, s_space, axis=0, edge_order=2)
    u_sgrad_s1 = np.gradient(i_us, s_space, axis=0, edge_order=2)
    u_phigrad_s1 = np.gradient(i_uphi, s_space, axis=0, edge_order=2) 
    u_zgrad_s1 = np.gradient(i_uz, s_space, axis=0, edge_order=2)
    

    tempgrad_phi1 = np.gradient(i_temp, phi_space, axis=1, edge_order=2)
    u_sgrad_phi1 = np.gradient(i_us, phi_space, axis=1, edge_order=2)
    u_phigrad_phi1 = np.gradient(i_uphi, phi_space, axis=1, edge_order=2)
    u_zgrad_phi1 = np.gradient(i_uz, phi_space, axis=1, edge_order=2)
    
    tempgrad_z1 = np.gradient(i_temp, z_space, axis=2, edge_order=2)
    u_sgrad_z1 = np.gradient(i_us, z_space, axis=2, edge_order=2)
    u_phigrad_z1 = np.gradient(i_uphi, z_space, axis=2, edge_order=2)
    u_zgrad_z1 = np.gradient(i_uz, z_space, axis=2, edge_order=2)
    
    # flatten gradients
    tempgrad_s = tempgrad_s1.reshape(len(s) * len(phi) * len(z))
    u_sgrad_s = u_sgrad_s1.reshape(len(s) * len(phi) * len(z))
    u_phigrad_s = u_phigrad_s1.reshape(len(s) * len(phi) * len(z))
    u_zgrad_s = u_zgrad_s1.reshape(len(s) * len(phi) * len(z))
    
    tempgrad_phi = tempgrad_phi1.reshape(len(s) * len(phi) * len(z))
    u_sgrad_phi = u_sgrad_phi1.reshape(len(s) * len(phi) * len(z))
    u_phigrad_phi = u_phigrad_phi1.reshape(len(s) * len(phi) * len(z))
    u_zgrad_phi = u_zgrad_phi1.reshape(len(s) * len(phi) * len(z))
    
    tempgrad_z = tempgrad_z1.reshape(len(s) * len(phi) * len(z))
    u_sgrad_z = u_sgrad_z1.reshape(len(s) * len(phi) * len(z))
    u_phigrad_z = u_phigrad_z1.reshape(len(s) * len(phi) * len(z))
    u_zgrad_z = u_zgrad_z1.reshape(len(s) * len(phi) * len(z))
    
    return tempgrad_s, u_sgrad_s, u_phigrad_s, u_zgrad_s, tempgrad_phi, u_sgrad_phi, u_phigrad_phi,            u_zgrad_phi, tempgrad_z, u_sgrad_z, u_phigrad_z, u_zgrad_z
    


# In[15]:


def cyl2cart(s, phi):
    x = s * np.cos(phi)
    y = s * np.sin(phi)
    return x, y


# In[16]:


def merge_two_dicts(x, y):
    z = x.copy()   
    z.update(y)    
    return z


# In[17]:


def create_metadata(coreData, KE, Re, Nu):
    metadata1 = coreData.__dict__
    metadata2 = {'Kinetic energy': KE, 'Reynolds number': Re, 'Nu': Nu} 
    metadata = merge_two_dicts(metadata1, metadata2)
    return metadata


# In[18]:


def integrand_Nu(u_r, temp):
    integrand = u_r * temp
    return integrand


# ### Find global Nusselt number
# 
# $C = \frac{ \langle u_r \rangle}{A_{cmb}} \int^{r_o}_{r_i} \langle T \rangle\ dV$
# 
# $ Nu = 1 + \frac{C}{\Delta T}$

# In[19]:


def Nu_sph_harm(temp, u_r, r, az, el, r1, az1, el1, cmb_mask, icb_mask):
    temp_flat = temp.flatten()
    temp_cmb1 = temp_flat[cmb_mask]
    temp_icb1 = temp_flat[icb_mask]
    temp_cmb = temp_cmb1.reshape(len(az), len(el))
    temp_icb = temp_icb1.reshape(len(az), len(el))
    
    # find average temperature on each shell using spherical harmonics
    az2, el2 = np.meshgrid(az, el)
    az3 = az2.flatten()
    el3 = el2.flatten()
    m = 3
    l = 100
    temp_cmb_sph = temp_cmb1 * sph_harm(m, l, az3, el3)
    temp_icb_sph = temp_icb1 * sph_harm(m, l, az3, el3)

    # Flatten the spherical harmonics arrays
    temp_cmb_sph_flat = temp_cmb_sph.flatten()
    temp_icb_sph_flat = temp_icb_sph.flatten()

    # find delta T
    delta_temp_sph = np.abs(temp_cmb_sph_flat - temp_icb_sph_flat)

    deltaT = delta_temp_sph.reshape(( len(az), len(el)))
    
    # find int delta T
    int_deltaT = np.trapz(np.trapz(deltaT, el, axis=1), az, axis=0) 
    
    # find wt using the function
    integrand_wT = integrand_Nu(u_r, temp)
    
    # Find convective heat transfer
    convective_ht_trapz = np.trapz(np.trapz(np.trapz(integrand_wT, el, axis=2), az, axis=1), r, axis=0) # as for int_wt
    
    # surface area of outer shell
    total_surf_cmb = 4 * np.pi * r.max()**2
    
    convection_trapz = 1/total_surf_cmb * convective_ht_trapz
    Nu_trapz2 = 1 + convection_trapz/int_deltaT
    
    return Nu_trapz2


# ### Find Turbulent Kinetic Energy (TKE) = 0.5 * (u'^2 + v'^2 + w'^2)
# 
# $u_r^{\prime} = u_r - \langle u \rangle $
# 
# $u_{el}^{\prime} = u_{el} - \langle u \rangle $
# 
# $u_{az}^{\prime} = u_{az} - \langle u \rangle $

# In[20]:


def TKE(u_r, u_el, u_az):
    ur_mean = np.nanmean(u_r)
    uel_mean = np.nanmean(u_el)
    uaz_mean = np.nanmean(u_az)
    
    ur_prime = u_r - ur_mean
    uel_prime = u_el - uel_mean
    uaz_prime = u_az - uaz_mean
    
    TKE = 0.5 * (ur_prime**2 + uel_prime**2 + uaz_prime**2)
    TKE_mean = np.nanmean(TKE)
    print("Mean TKE = ", TKE_mean)
    
    return TKE_mean


# # Find turbulent intensity
# 
# Turbulent Intensity ($u_r$): $I = \frac{u_{r(RMS)}^{\prime}}{\langle u_r \rangle} * 100$
# 
# Turbulent Intensity ($u_{el}$): $I = \frac{u_{el(RMS)}^{\prime}}{\langle u_{el} \rangle} * 100$
# 
# Turbulent Intensity ($u_{az}$): $I = \frac{u_{az(RMS)}^{\prime}}{\langle u_{az} \rangle} * 100$

# In[21]:


def turb_I(u_r, u_el, u_az, temp):
    
    ur_mean = np.nanmean(u_r)
    uel_mean = np.nanmean(u_el)
    uaz_mean = np.nanmean(u_az)
    temp_mean = np.nanmean(temp)
    
    ur_prime = u_r - ur_mean
    uel_prime = u_el - uel_mean
    uaz_prime = u_az - uaz_mean
    temp_prime = temp - temp_mean
    
    ur_prime_rms = np.sqrt(np.nanmean(ur_prime**2))
    uel_prime_rms = np.sqrt(np.nanmean(uel_prime**2))
    uaz_prime_rms = np.sqrt(np.nanmean(uaz_prime**2))
    temp_prime_rms = np.sqrt(np.nanmean(temp_prime**2))
    
    TI_ur = (ur_prime_rms / abs(ur_mean)) * 100
    TI_uel = (uel_prime_rms / abs(uel_mean)) * 100
    TI_uaz = (uaz_prime_rms / abs(uaz_mean)) * 100
    TI_temp = (temp_prime_rms / abs(temp_mean)) * 100
    
    return TI_ur, TI_uel, TI_uaz, TI_temp


# In[22]:


def analyse_timestep():
    start_time = time.process_time()
    
    # settings
    input_path, output_path, num = basic_settings()
    input_list = create_input_list(input_path)
    norm_path = os.path.normpath(output_path)
    timesteps = []
    results = "output_data"
    filepath_new1 = os.path.join(output_path, results)
    print("Output folder path: ", filepath_new1)
    
    # Create file for all outputs
    date = datetime.now()
    folder =  "Folder" + date.strftime('_%Y_%m_%d') # put name in here
    filepath_new = os.path.join(filepath_new1, folder)
    
    try:
        os.makedirs(filepath_new, exist_ok = False)
        print("Directory '%s' created" %folder)
    except OSError as error: # test this on linux - laptop just overwrites
        print("Directory already exists")
        folder2 = "Folder" + date.strftime('_%Y_%m_%d_') + "duplicate"
        filepath_new = os.path.join(filepath_new1, folder2)
        os.makedirs(filepath_new, exist_ok = True) 
    print("Folder name:", folder2)
    
    for i in range(0, len(input_list)):
        
        # load timestep to be processed 
        timesteps.append(os.path.join(input_path, input_list[i]))
    
        # extract data from netCDF file
        coreData = nc.Dataset(timesteps[i])
        
        # bring out variables
        el, r, az, temp, u_r, u_el, u_az, temp_grad_el, temp_grad_r = create_variables(coreData)
        
        #Turbulent kinetic energy and turbulent intensity
        TKE_mean = TKE(u_r, u_el, u_az) # not stored
        TI_ur, TI_uel, TI_uaz, TI_temp = turb_I(u_r, u_el, u_az, temp) # not stored
        
        # create output array 
        s, phi, z = output_grid(r, num)  #cylinder
        #x, y, z = output_grid(r, num)  # cube
        
        # interpolate onto even grid
        i_temp, i_ur, i_uel, i_uaz, rr, pp, tt, r_check, az_check, el_check = interpolation_even_grid(el, r, az, temp, u_r,                                                                                                       u_el, u_az)
        # spherical shell values for KE, Re, Nu
        r1, az1, el1, cmb_mask, icb_mask = create_grid(r, az, el)
        KE_trapz, Re_traps, KE, Re = KE_Re(r1, az1, el1, u_r, u_az, u_el, r, az, el) # simpsons values for checks
        
        print("KE = ", KE)
        print("Re = ", Re)
        
        # global Nusselt number
        Nu = Nu_sph_harm(temp, u_r, r, az, el, r1, az1, el1, cmb_mask, icb_mask)
        print("Nu = ", Nu)
        
        # interpolate temp and velocity components
        ss_tc, pp_tc, zz_tc, u_s, u_phi, u_z, rr_tc, tt_tc, i_ur_tc, i_uaz_tc, i_uel_tc,                  i_temp_tc, TC_curved_surface, TC_CMB, TC_ICB = interpolation_TC_grid(num, r, az, el, temp, u_r, u_el, u_az, s, phi, z)
        
        
        # gradients - using even grid
        tempgrad_s, u_sgrad_s, u_phigrad_s, u_zgrad_s, tempgrad_phi, u_sgrad_phi, u_phigrad_phi,         u_zgrad_phi, tempgrad_z, u_sgrad_z, u_phigrad_z, u_zgrad_z = gradients(s, phi, z, i_temp_tc,                                                                                u_s, u_phi, u_z)
        
        # create cartesian coordinates - for plots
        x_plot, y_plot = cyl2cart(ss_tc, pp_tc)  #comment this out if generating a cube in the first instance
        
        # metadata
        metadata = create_metadata(coreData, KE, Re, Nu)
              
        # export metadata csv file
        filename2 = "metadata_" + input_list[i][5:-38] + "_#%s.csv" % str(i+1)
        filepath2 = os.path.join(filepath_new, filename2)
        with open(filepath2, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(metadata.keys())
            writer.writerow([format(x, '.15f') for x in metadata.values()])
        
        # parquet: TC data
        filename3 = input_list[i][5:-38] + "_TC.parquet" + "%s" % str(i+1)
        filepath3 = os.path.join(filepath_new, filename3)
        
        pa_ss_tc = pa.array(ss_tc)
        pa_pp_tc = pa.array(pp_tc)
        pa_zz_tc = pa.array(zz_tc)
        pa_i_temp_tc = pa.array(i_temp_tc)
        pa_u_s = pa.array(u_s)
        pa_u_phi = pa.array(u_phi)
        pa_u_z = pa.array(u_z)
        pa_tempgrad_s = pa.array(tempgrad_s)
        pa_u_sgrad_s = pa.array(u_sgrad_s)
        pa_u_phigrad_s = pa.array(u_phigrad_s)
        pa_u_zgrad_s = pa.array(u_zgrad_s)
        pa_tempgrad_phi = pa.array(tempgrad_phi)
        pa_u_sgrad_phi = pa.array(u_sgrad_phi)
        pa_u_phigrad_phi = pa.array(u_phigrad_phi)
        pa_u_zgrad_phi = pa.array(u_zgrad_phi)
        pa_tempgrad_z = pa.array(tempgrad_z)
        pa_u_sgrad_z = pa.array(u_sgrad_z)
        pa_u_phigrad_z = pa.array(u_phigrad_z)
        pa_u_zgrad_z = pa.array(u_zgrad_z)
        pa_curved_surface = pa.array(TC_curved_surface)
        pa_CMB = pa.array(TC_CMB)
        pa_ICB = pa.array(TC_ICB)
        pa_x = pa.array(x_plot)
        pa_y = pa.array(y_plot)
        pa_r = pa.array(rr_tc)
        pa_el = pa.array(tt_tc)
        
        #create a dictionary mapping headers to arrays
        tangent_cylinder = {'s': pa_ss_tc, 'phi': pa_pp_tc, 'z': pa_zz_tc, 'curved_surface': pa_curved_surface,                            'TC_CMB': pa_CMB, 'TC_ICB': pa_ICB, 'temp': pa_i_temp_tc, 'u_s': pa_u_s, 'u_phi': pa_u_phi,                            'u_z': pa_u_z, 'tempgrad_s': pa_tempgrad_s, 'u_sgrad_s':                            pa_u_sgrad_s, 'u_phigrad_s': pa_u_phigrad_s, 'u_zgrad_s': pa_u_zgrad_s,                             'tempgrad_phi': pa_tempgrad_phi, 'u_sgrad_phi': pa_u_sgrad_phi, 'u_phigrad_phi':                            pa_u_phigrad_phi, 'u_zgrad_phi': pa_u_zgrad_phi, 'tempgrad_z': pa_tempgrad_z,                            'u_sgrad_z': pa_u_sgrad_z, 'u_phigrad_z': pa_u_phigrad_z, 'u_zgrad_z': pa_u_zgrad_z,                            'x_plot': pa_x, 'y_plot': pa_y, 'r': pa_r, 'theta': pa_el }
        
        # create a table from these
        table = pa.Table.from_arrays( [ pa_ss_tc, pa_pp_tc, pa_zz_tc, pa_curved_surface, pa_CMB, pa_ICB, pa_i_temp_tc,                                    pa_u_s, pa_u_phi, pa_u_z,                                        pa_tempgrad_s, pa_u_sgrad_s, pa_u_phigrad_s, pa_u_zgrad_s, pa_tempgrad_phi,                                      pa_u_sgrad_phi, pa_u_phigrad_phi, pa_u_zgrad_phi, pa_tempgrad_z,                                       pa_u_sgrad_z, pa_u_phigrad_z, pa_u_zgrad_z, pa_x, pa_y, pa_r, pa_el ], ['s', 'phi',                                       'z',                                     'curved_surface', 'TC_CMB', 'TC_ICB',  'temp', 'u_s', 'u_phi', 'u_z', 'tempgrad_s',                                    'u_sgrad_s',                                    'u_phigrad_s', 'u_zgrad_s', 'tempgrad_phi', 'u_sgrad_phi', 'u_phigrad_phi',                                    'u_zgrad_phi', 'tempgrad_z', 'u_sgrad_z', 'u_phigrad_z', 'u_zgrad_z', 'x_plot',                                    'y_plot', 'r', 'theta' ])
        
        # write to parquet file
        pq.write_table(table, filepath3)
        
        print("Number %s complete" % str(i+1))
        print("")
    end_time = time.process_time()
    cpu_time = end_time - start_time
    print("CPU time taken: ", cpu_time)
    
analyse_timestep()


# In[ ]:





# In[ ]:





# In[ ]:





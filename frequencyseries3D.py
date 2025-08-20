import numpy as np
from kuibit import timeseries
from kuibit.series import BaseSeries, sample_common
import h5py
import matplotlib.pyplot as plt
from kuibit import timeseries as ts
from kuibit import series
from kuibit import unitconv as uc
from kuibit.gw_utils import luminosity_distance_to_redshift
import urllib.parse
import os
from kuibit.simdir import SimDir
from kuibit.grid_data import UniformGrid
import kuibit.grid_data as gd
from scipy.interpolate import interp1d
from scipy.signal import welch
import numpy as np
import pandas as pd

numData = 3266352
iteration = 1225
numFiles = 128
filenames = []

# Step 1: Access the data from your HDF5 file

# Adds in the filenames to look for
def extractData(datasetName, numberFiles, iterations, sRate, resolution):

    directory = "C://Users//Auyus//Research//Data//" + datasetName
    # Step 1: Access the data from your HDF5 file
    gf = SimDir(directory).gf

    print("Available keys in gf.xyz:", list(gf.xyz.keys()))

    temp_out_frac = gf.xyz['temp_out_frac']

    os.chdir(directory)
    
    filenames = []

    for x in range(numberFiles):

        filenames.append(f"temp_out_frac.file_{x}.h5")



    all_distances = []

    all_temperatures = []

    data = []


    # Step 2: Extract the data

    for file_index, filename in enumerate(filenames):

        distances = []
        
        temperatures = []
        



        with h5py.File(filename, 'r') as file:
        
            # Iterate through the dataset names
            dataset_name = f"MHD_ANALYSIS::temp_out_frac it={iterations} tl=0 rl=0 c={file_index}"
            
        
            # Access the dataset
            dataset = file[dataset_name]
            
            # Read data from the dataset

            print(f"Shape of data {dataset_name}: {dataset.shape}")
            dataset_squeezed = np.squeeze(dataset)
            dataset_flattened = dataset_squeezed.flatten()
            print(dataset_flattened.shape)
            
            #Append command, and print statement for confirmation
            
            data.append(dataset_flattened)
            print(len(data))
            
            
            
            
        
            
            
            
            
        # Step 3: Data Processing Code - You may notice several print statements - these were all statements I made as I built my code to ensure things were going smoothly
    all_data = np.concatenate(data, axis=0)


                    
        #distances = temp_out_frac.available_times
            
        #print(it_distances)



    #distances.append(it_distances)

    print(all_data)

    #total_data_points = sum(len(data) for data in all_data)

    #print("Total data points:", total_data_points)

    print(len(all_temperatures))
            

    distance_indices = np.array(distances).flatten()

    #temperature_indices = np.array(all_temperatures).flatten() - these needed to be comented out as I updated my code and added these features beforehand

    #temperature_indices2 = np.asarray(all_temperatures).squeeze()

    temperature_indices2 = all_data * (3*(10**6))

    print(temperature_indices2)

    print(len(temperature_indices2))

    print(min(temperature_indices2))

    print(max(temperature_indices2))

    new_distance_indices = np.linspace((-4.629e+26), (4.629e+26), numData) # creates period indices that would later become wavenumber values after the fourier transform

    print(np.array(distance_indices).shape)

    print(np.array(all_data).shape)


    temp_abs = np.abs(temperature_indices2) 

    print(min(temp_abs))

    print(max(temp_abs))

    #gw = ts.TimeSeries(new_distance_indices, temp_abs)

    # These values were what I thought I needed to use for grid space calculations before I realized the error in my code - it wasnt appending the data correctly
    '''
    c = (3*(10**8))

    dx = 1.6924*(10**24)

    constt = 1*(10**13)

    rad_const = 0.03966657386126

    dx = 1.6924*(10**24)*180
    dx = (dx / rad_const) * (4.75*10**-16)

    def freq_to_l(frequency_data):
        wavenumbers = ((2 * np.pi * frequency_data)/c)*dx 
        return wavenumbers






    # This was my old method of creating a graph - the uncommented version below is more efficient
    def plot(ser, lab1="d h", lab2="t", *args, **kwargs):
            plt.ylabel(lab1)
            plt.xlabel(lab2)
            plt.plot(ser, *args, **kwargs)
            
    gw_FrequencySeries = gw.to_FrequencySeries()    
            
    plot(gw_FrequencySeries)       
                
    plt.figure()

    plt.title('Temperature Variatons in the CMB')

    plt.xlabel('Frequency')

    plt.ylabel('Temperature Variation')    


    plt.savefig("C://Users//Auyus//Temp Pets//frequencyseries.png")

    plt.show()



    '''
    # This Is what creates the graph for the PSD! (The Fourier Transform) - fs is the sampling frequency, 
    #and nperseg is the resolution, the higher the value the higher the amplitudes
    frequencies, psd = welch(temp_abs, fs=(sRate), nperseg=resolution)

    #wavenumber_values = freq_to_l(frequencies)

    #Sorts out real and imaginary values, squares them, then adds both to create the angular power spectrum values
    psd_real = np.real(psd)

    psd_imag = np.imag(psd)

    aps = (psd_real ** 2) + (psd_imag ** 2)

    return frequencies, aps




datasets = { "A": "TP Base Model\n", "B": "TP Minus10 EW\n", "C": "TP Plus10 EW\n", "D": "TP No EW\n", "E": "TP 64 Res\n", "F": "TP 32 Res\n", "G": "TP Error Calculation\n" } 

print("A. TP Base Model\nB. TP Minus10 EW\nC. TP Plus10 EW\nD. TP No EW\nE. TP 64 Res\nF. TP 32 Res\nG. TP Error Calculation\n")

print("Which Datasets? (e.g., A, B, C)") 

choices = input("Choose datasets (e.g., A,B,C): ").split(',') 

frequencies, psd = [], []

labels = []

for choice in choices:

    samplingRate = 16384

    res = 1024

    if choice == "A":
        datasetName = "TP Base Model"
    elif choice == "B":
        datasetName = "TP Minus10 PT"
    elif choice == "C":
        datasetName = "TP Plus10 PT"
    elif choice == "D":
        datasetName = "TP No PT"
    elif choice == "E":
        datasetName = "TP 64 Res"
        iteration = 1200
        numFiles = 64
        samplingRate /= 4
        res /= 4
    elif choice == "F":
        datasetName = "TP 32 Res"
        iteration = 1200
        numFiles = 32
        samplingRate /= 16
        res /= 16
    elif choice == "G":
        datasetName = "TP Error Calculation"
    else:
        raise ValueError("Invalid Dataset")
    frequencies, psd = extractData(datasetName, numFiles, iteration, samplingRate, res)
    plt.semilogy(frequencies, psd, label = datasetName)
    labels.append(datasetName)
    
observed_data = np.loadtxt('https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/cosmoparams/COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt', skiprows=1)

print(observed_data.shape)

l_observed = observed_data[:, 0].flatten()  # Multipole moments for observed data

C_l_observed = observed_data[:, 1].flatten()  # Power spectrum values for observed data


        #Plots and displays graph

print("Compare to observed power spectrum?")

if input() == "y":
        plt.semilogy(l_observed, C_l_observed, label = "Observed")

plt.xlabel('l (Wavenumber)')

plt.xscale('log')

plt.ylabel('Temperature (uK^2)')

plt.title('Power Spectral Density')

plt.legend(loc="lower left")

plt.show()
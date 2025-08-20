from kuibit.simdir import SimDir
import kuibit.grid_data as gd

SCALE_FACTOR = 1.02

directory = "/mnt/c/Users/Auyus/Research/Data/VIZ_128_GR"
    # Step 1: Access the data from your HDF5 file
gf = SimDir(directory).gf

rho_out_frac = gf.xyz['rho_out_frac']

print("Available times:", rho_out_frac.available_iterations)


raw_data = rho_out_frac[1250]  # Get the data at time 1225

print("Raw data shape:", raw_data.shape)
print("Raw data max:", raw_data.max())  # No axis argument
print("Raw data min:", raw_data.min())  # No axis argument

print("Lower bound (x0):", raw_data.x0)
print("Upper bound (x1):", raw_data.x1)

grid = gd.UniformGrid([132, 132, 132], x0 = raw_data.x0, x1 = raw_data.x1)

my_data = rho_out_frac.read_on_grid(1250, grid, False)

multiplier = 


print(f"The absolute maximum is {my_data.abs_max()}")
print(f"The absolute maximum occurs at {my_data.coordinates_at_maximum()}")
print(f"Early universe Conversion: {my_data.coordinates_at_maximum() * 3.93105856e-16}")


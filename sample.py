from inductor import *

# Load in the s2p data.
inductor = Inductor(data="sample.s2p")

# Print the modeled parameters.
print(inductor.model_parameters)

# Write out the data in spice format.
inductor.write_spice(filename='./my_inductor.cir')

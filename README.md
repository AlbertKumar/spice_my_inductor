# spice_my_inductor
Takes an s-parameter (Touchstone s2p) file and creates a lumped component Spice (.sp) model.

1. Download the files:
  - inductor.py
  - skrf_extensions.py
  - aux.py

2. Make sure you have the following installed.
  - numpy
  - scikit-rf
  - lmfit
  - matplotlib
  
Example script.
```
from inductor import *

# Load in the s2p data.
inductor = Inductor(data="sample.s2p")

# Print the modeled parameters.
print(inductor.model_parameters)

# Write out the data in spice format.
inductor.write_spice(filename='./my_inductor.cir')
```

References:
H. -H. Chen, H. -W. Zhang, S. -J. Chung, J. -T. Kuo and T. -C. Wu, "Accurate Systematic Model-Parameter Extraction for On-Chip Spiral Inductors," in IEEE Transactions on Electron Devices, vol. 55, no. 11, pp. 3267-3273, Nov. 2008, doi: 10.1109/TED.2008.2005131.

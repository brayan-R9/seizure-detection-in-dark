import numpy as np

patches_file = "./data/patches/patches_0.npy"  # Example file
patches = np.load(patches_file)
print("Patches Shape:", patches.shape)
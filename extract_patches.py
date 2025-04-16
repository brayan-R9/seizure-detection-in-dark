# extract patches based on keypoints

'''import math, cv2
import numpy as np

def norm(x):
    y = ((x - np.min(x)) / (np.max(x) - np.min(x))) * 255
    return y

def gen_kernel(size, sigma):
    kernel = np.fromfunction(lambda x, y: (1/(2*math.pi*sigma**2)) * math.e ** ((-1*((x-(size-1)/2)**2+(y-(size-1)/2)**2))/(2*sigma**2)), (size, size))
    kernel = kernel / np.sum(kernel)
    kernel = (kernel - np.min(kernel)) / (np.max(kernel) - np.min(kernel))
    return kernel

def extract_patches(img, kpts, kernel_size = 128, kernel_sigma=0.3, scale=1/4):
    img_shape = img.shape # 1080 x 1920 x 3 video resolution
    pad_img = np.zeros((img_shape[0]+kernel_size*2, img_shape[1]+kernel_size*2, 3))
    pad_img[kernel_size:-kernel_size, kernel_size:-kernel_size, :] = img
    kernel = gen_kernel(kernel_size,kernel_size*kernel_sigma)
    kernel = np.expand_dims(kernel,2).repeat(3,axis=2)
    kpts = np.delete(kpts, [[1],[-3],[-4]], axis=0) # 
    patches = np.zeros((15,math.ceil(scale*kernel_size),math.ceil(scale*kernel_size),3))
    for idx in range(15):
        tmp = norm(pad_img[int(kpts[idx,1]+0.5*kernel_size):int(kpts[idx,1]+1.5*kernel_size), int(kpts[idx,0]+0.5*kernel_size):int(kpts[idx,0]+1.5*kernel_size),:] * kernel)
        tmp = cv2.resize(tmp, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        patches[idx,:,:,:] = tmp
    return patches'''

# extract patches based on keypoints
import math, cv2
import numpy as np

def norm(x):
    y = ((x - np.min(x)) / (np.max(x) - np.min(x))) * 255
    return y

def gen_kernel(size, sigma):
    kernel = np.fromfunction(lambda x, y: (1/(2*math.pi*sigma**2)) * math.e ** ((-1*((x-(size-1)/2)**2+(y-(size-1)/2)**2))/(2*sigma**2)), (size, size))
    kernel = kernel / np.sum(kernel)
    kernel = (kernel - np.min(kernel)) / (np.max(kernel) - np.min(kernel))
    return kernel

def extract_patches(img, kpts, kernel_size=128, kernel_sigma=0.3, scale=1/4):
    img_shape = img.shape  # 1080 x 1920 x 3 video resolution

    # Pad the image
    pad_img = np.zeros((img_shape[0] + kernel_size * 2, img_shape[1] + kernel_size * 2, 3))
    pad_img[kernel_size:-kernel_size, kernel_size:-kernel_size, :] = img

    # Generate the kernel
    kernel = gen_kernel(kernel_size, kernel_size * kernel_sigma)
    kernel = np.expand_dims(kernel, 2).repeat(3, axis=2)

    # Validate keypoints
    if kpts.shape[0] < 18:
        #print(f"Warning: Found fewer than 18 keypoints. Padding with zeros.")
        padded_kpts = np.zeros((18, 2))
        padded_kpts[:kpts.shape[0], :] = kpts
        kpts = padded_kpts

    # Ensure the keypoints are within bounds
    kpts = np.clip(kpts, 0, max(img_shape[0], img_shape[1]))

    # Safely delete indices only if the size allows
    if kpts.shape[0] >= 18:
        try:
            kpts = np.delete(kpts, [[1],[-3],[-4]], axis=0)  # Ensure it works with 18 keypoints
        except IndexError:
            print(f"Keypoints deletion failed. Keypoints shape: {kpts.shape}")
            return None

    # Initialize patches
    patches = np.zeros((15, math.ceil(scale * kernel_size), math.ceil(scale * kernel_size), 3))

    # Extract patches
    for idx in range(15):
        try:
            x_min = int(kpts[idx, 0] + 0.5 * kernel_size)
            x_max = int(kpts[idx, 0] + 1.5 * kernel_size)
            y_min = int(kpts[idx, 1] + 0.5 * kernel_size)
            y_max = int(kpts[idx, 1] + 1.5 * kernel_size)

            # Ensure the patch is within bounds
            if x_min < 0 or y_min < 0 or x_max > pad_img.shape[1] or y_max > pad_img.shape[0]:
                print(f"Keypoint {idx} is out of bounds: {kpts[idx]}")
                continue

            # Apply kernel and resize
            tmp = norm(pad_img[y_min:y_max, x_min:x_max, :] * kernel)
            tmp = cv2.resize(tmp, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            patches[idx, :, :, :] = tmp
        except Exception as e:
            print(f"Error processing keypoint {idx}: {e}")

    return patches

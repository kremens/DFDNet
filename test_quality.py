from PIL import Image
import numpy as np

path_1 = '/home/ubuntu/DFL_crops/DFL_ALIGNED_RAWPRED/merge_rawpred_enchanced_r0_compression_0/int0010_Sassy_aligned_wf_v03.0003.png'
path_2 = '/home/ubuntu/DFL_crops/DFL_ALIGNED_RAWPRED/merge_rawpred_enchanced_r0_compression_6/int0010_Sassy_aligned_wf_v03.0003.png'


image_1 = np.array(Image.open(path_1))
image_2 = np.array(Image.open(path_2))

print(np.sum(abs(image_1 - image_2)))

PIL_image = Image.fromarray(abs(image_1 - image_2)).convert('RGB')

PIL_image.show()
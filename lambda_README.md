# Lambda Notes


### Install

Step 1:

```
su craig
sudo su
conda create -n DFD-deepvoodoo python=3.6
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch -n DFD-deepvoodoo
conda install -c conda-forge/label/cf202003 dlib -n DFD-deepvoodoo

conda activate DFD-deepvoodoo
pip install dominate
pip install opencv-python
pip install scikit-image
pip install tqdm
```

Step 2:

Download these files [links](https://drive.google.com/drive/folders/1bayYIUMCSGmoFPyd4Uu2Uwn347RW-vl5).

Unzip and put them to DFDNet root directory.


### Usage

Step 1: Copy test images to somewhere `path-to-input-images` on your machine.

Step 2: Run the following command on any tweeks 

```
conda activate DFD-deepvoodoo

python /ParkCounty/home/SharedApp/DeepFaceLab_Linux/DFDNet_DeepVooDoo/test_FaceDict.py  \
--test_name name-of-the-character \
--test_dir path-to-input-images \
--results_dir path-to-results \
--gpu_id 0 \
--upscale your-scale \
--only_final 

```

Example

```
conda activate DFD-deepvoodoo

python /ParkCounty/home/SharedApp/DeepFaceLab_Linux/DFDNet_DeepVooDoo/test_FaceDict.py  \
--test_name deepvoodoo \
--test_dir ~/1-IMPORT \
--results_dir ~/2-RESULTS \
--gpu_id 0 \
--upscale 4 \
--only_final 
```


* `test_name`: Name of the character. The input images are expected to be in `test_dir/test_name`, the output results will be stored in `results_dir/test_name`. Default: `deepvoodoo`
* `test_dir`: Folder to store subfolders of test images. Default: `~/1-IMPORT`
* `results_dir`: Folder to store subfolders of results. Default: `~/2-RESULTS`
* `gpu_id`: Index of GPU. Default 
* `upscale`: Upscale factor. The input crops are resized to `512`. The output result will be `512`x`upscale`
* `only_final`: Add this flag to only save results of Step3 and Step4.

### Results

 <table  style="float:center" width=100%>
 <tr>
  <th><B> Input </B></th><th><B>Final Results (UpScaleWhole=4)</B></th>
 </tr>
  <tr>
  <td>
  <img src='./Imgs/parker_input.jpg' width="512">
  </td>
  <td>
  <img src='./Imgs/parker_output.jpg' width="512">
  </td>
 </tr>
  
 </table>


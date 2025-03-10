# Sunspots Finder

Two approached were considered : the first one consisted in thresholding the intensity of the image of the sun to find sunspots as a mask, then listing them and computing their areas.
The .py files are the ones that matter if you want get started fast. The notebooks in the Methods folders are drafts.  

In both methods, the output will be the sunspots in terminal and an image with visual markers in the results folder (the output folder can be changed).


### Scraping data

Data images are used for two purposes : testing the methods with examples and training method 2. Images come from NASA, they are HMI Intensitygrams, flattened, orange in 1024x1024 format. Only January 2025 images were scraped for this project. You can access them for free at https://sdo.gsfc.nasa.gov/assets/img/browse/2025/01/.  
Note that no labels are given with the images. The performances are left to the appreciation of the users, and the AI techniques must be unsupervised.  

The notebook scrap.ipynb can be executed to download all relevant images to the image folder, though it is advised to only download a few days of data if it is for demonstration.
Requires bs4.  


### Method 1

#### Get started

Execute:

``` 
python Method1/method1.py --input_path 'images\20250110_164500_1024_HMIIF.jpg'
```

You can also add optional arguments output_folder, threshold_ratio and min_area:

```
python Method1/method1.py --input_path 'images\20250110_164500_1024_HMIIF.jpg' --output_folder 'Method1/results' --threshold_ratio 0.88 --min_area 7
```

#### Description

A threshold is determined, proportionally to the average intensity of the sun on the image (taking into account only pixels bright enough). A coefficient of 0.88 was found to provide the best results, and is the default value but can be changed as an optional argument.
The proportional thresholds guarantees robustness to a certain point : even if the image is overall less bright for some reason (instruments changing or special solar conditions) the method should work.  
Then the connected components are determined, and the holes are filled. Only spots large enough are considered, so the method is not sensible to noise. The minimum area is 7 by default, as it was found to yield the best results, but it is also an optional argument that can be changed when calling the method.  
The borders of these components are also determined, as to make visual markers for verification. The image with markers is stored in the results folder.

#### Known limitation

If a sunspot is on the very edge of the sun, it possibly can be ignored. This can be fixed easily with skImage segmentation tools, for example chan_vese with a very high mu value can delimit the sun very well. I didn't have time to implement this.

### Method 2

#### Get started
Execute:

```
python Method2/method2.py --input_path 'images\20250110_164500_1024_HMIIF.jpg'
```

Optional arguments may be used:

```
python Method2/method2.py --input_path 'images\20250110_164500_1024_HMIIF.jpg' --output_folder 'Method2/results/' --threshold 0.1 --min_area 3
```


#### Description

The straightforward method 1 seems to work well, but comparing it with deep learning techniques might reveal insights.   
This method implements a convolutional neural network aimed at a segmentation task. It produces a binary mask that detects sunspots. The architecture uses classical components of CNNs, and is not very deep (2 encooding and 2 decoding layers) because the task is a priori not too complex.  

There are two encoding layers, then two decoding layers. The encoding layers are composed of a convolution kernel that produces a map of 64 features for the first encoding and 128 for the second, followed by batch normalization (a classical trick to stabilize learning), folowed by a ReLU activation and a max pooling on square windows of size 2x2. 
After that, the decoder uses bilinear interpolation to upsample the low-res feature map, reduces back the number of features with a convolution kernel, applies batch normalization and ReLU. A final convolution yields a single-channel output, that is passed to a sigmoid to get a number between 0 and 1.  
Later, a threshold is applied (by default 0.1 but can be changed, the higher the threshold the less spots will be detected) to determine if a pixel is a sunspot or not.  
Only spots of area greater than some value (by default 3 but can be changed) are kept.  

The model was unsupervisedly trained on scraped data from NASA.  

#### Known limitations and things to do
- For now, the model transforms the image to a 256x256 at the very beggining, meaning the result would not be very accurate for the computation of areas. This reduction was necessary to limit training time on my local computer, but can be fixed  

- Different hyperparameters should be tested (min_area and threshold)  
- The model should be trained on more images, with different colors and intensities for better robustness.
- We should provide a detailed comparison with method 1
- Method1 can be useful to label data so that a supervised CNN model may be used instead. 

## Requirements
Code was executed with the following versions:

MatPlotLib 3.5.0, Numpy 1.26.4, Scikit-Image 0.21.0, CV2 4.7.0.72, PyTorch 2.5.1+cu118, PIL 11.1.0
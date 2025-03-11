import cv2
import numpy as np
import skimage.measure as skm
import argparse

def main():
    #Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_folder', type=str, default='Method1/results/')
    parser.add_argument('--threshold_ratio', type=float, default=0.88)
    parser.add_argument('--min_area', type=int, default=7)

    args = parser.parse_args()

    input_path, output_folder, threshold_ratio, min_area \
        = args.input_path, args.output_folder, args.threshold_ratio, args.min_area

    image_name = input_path.split("\\")[-1] 
    image_name = image_name.split('/')[-1] #get what is after the last /
    image_name = image_name.split('.')[0]

    # read the image in grayscale
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    image_array = np.array(image)

    # also import original image in color
    image_color = cv2.imread(input_path, cv2.IMREAD_COLOR)

    # Calculate mean intensity considering only pixels with intensity high enough
    bright_pixels = image_array[image_array > 25]
    mean_intensity = np.mean(bright_pixels)

    # Calculate threshold as alpha * mean_intensity
    # the threshold is proportional to the intensity of the sun for better robustness
    threshold = round(threshold_ratio * mean_intensity)

    # Create the mask of estimated sunspots
    mask = image_array < threshold
    mask = np.stack((mask,)*3, axis=-1)
    # color_mask = np.zeros_like(image_color)
    # color_mask[:,:,0] = mask
    # mask = color_mask

    #Get connected components of the mask = the sunspots
    labels = skm.label(mask, connectivity=2, background=0)
    regionprops = skm.regionprops(labels)

    #Filter the regions to keep only relevant ones
    scale_factor = (image_array.shape[0]/1024)**2
    regionprops = [regionprop for regionprop in regionprops if  10000 >= regionprop.area_filled/scale_factor >= min_area]

    # Create a mask for the borders
    border_mask = np.zeros_like(image_color)

    # Draw borders around each sunspot
    for i, region in enumerate(regionprops):
        if True or i % 2 == 0:  # Skip duplicate regions
            filled_area = region.filled_image
            minr, minc, _, maxr, maxc, _ = region.bbox
            
            # Create a temporary mask for this region
            temp_mask = np.zeros((maxr - minr + 1, maxc - minc + 1), dtype=bool)
            # Fill the mask with the filled area
            temp_mask[0:filled_area.shape[0], 0:filled_area.shape[1]] = filled_area[:,:,0]
            
            # Find the boundary pixels
            for x in range(temp_mask.shape[0]):
                for y in range(temp_mask.shape[1]):
                    if temp_mask[x, y]:
                        # Check if this pixel is on the boundary
                        if (x == 0 or x == temp_mask.shape[0]-1 or 
                            y == 0 or y == temp_mask.shape[1]-1 or
                            not temp_mask[x-1, y] or not temp_mask[x+1, y] or
                            not temp_mask[x, y-1] or not temp_mask[x, y+1]):
                            # Map back to original coordinates
                            border_mask[x + minr, y + minc] = [255, 255, 255] 

    # print(image_color.shape, border_mask.shape)
    # Superpose the borders on the original image
    border_superposed = cv2.addWeighted(image_color, 0.6, border_mask, 1, 0)

    # Display the analysis
    print(f'Number of sunspots: {len(regionprops)}')
    print('')
    for i in range(len(regionprops)):
        if True or i%2 == 0: #EDIT: nevermind
            print(f'Sunspot {i + 1}: {int(regionprops[i].area_filled)//scale_factor} pixels')

    # Save the resulting image. Uncomment to also save mask
    # cv2.imwrite(f'{output_folder}/mask_{image_name}.jpg', mask*255)
    cv2.imwrite(f'{output_folder}/output_{image_name}.jpg', border_superposed)
    print(f'Image with borders saved at {output_folder}/output_{image_name}.jpg')

if __name__ == '__main__':
    main()
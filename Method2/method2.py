import cv2
import numpy as np
import skimage.measure as skm
import argparse
import cv2
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision import transforms
import skimage.measure as skm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SunspotSegmenter(nn.Module):
    def __init__(self):
        super(SunspotSegmenter, self).__init__()
        # Encoder - extracts features
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Decoder that will generates segmentation mask
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()  # Output between 0 and 1, good for binary segmentation
        )

    def forward(self, x):
        # Two reducing layers
        x1 = self.enc1(x)
        x = self.pool1(x1)
        
        x2 = self.enc2(x)
        x = self.pool2(x2)
        
        # Then decoding layers
        x = self.upsample1(x)
        x = self.dec1(x)
        
        x = self.upsample2(x)
        x = self.dec2(x)
        
        return x
    
def detect_sunspots(model, image_path, device, threshold):
    img_name = image_path.split('\\')[-1]
    img_name = img_name.split('/')[-1]
    img_name = img_name.split('/')[-1]
    img_name = img_name.split('.')[0]

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image = Image.open(image_path).convert('RGB')
    original = image.copy()
    image = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        mask = model(image)
        mask = (mask > threshold).float()
        mask = mask.cpu().squeeze().numpy()
        mask = 255*mask
        
    # Calculate properties
    from skimage import measure
    labels = measure.label(mask)
    props = measure.regionprops(labels)
    
    return {
        'original': 255*image.squeeze().cpu().numpy().transpose(1, 2, 0),
        'mask': mask,
        'num_spots': len(props),
        'areas': [prop.area for prop in props],
        'centroids': [prop.centroid for prop in props],
        'image_name': img_name
    }

def analyze_sunspots(results, min_area, output_folder):
    print(output_folder)  
    # Get mask and original image
    mask = results['mask']
    original_image = np.array(results['original'])
    
    ## THIS FUNCTION IS BASICALLY THE SAME AS METHOD 1

    # Get connected components
    labels = skm.label(mask, connectivity=2, background=0)
    regionprops = skm.regionprops(labels)
    
    # Filter regions
    regionprops = [prop for prop in regionprops if 
                   10000 >= prop.area * 1024/original_image.shape[0] >= min_area] #renormalize area
    
    # Create border mask
    border_mask = np.zeros_like(original_image)
    
    # Draw borders
    for region in regionprops:
        filled_area = region.filled_image
        minr, minc, maxr, maxc = region.bbox
        
        # Create temporary mask
        temp_mask = np.zeros((maxr - minr, maxc - minc), dtype=bool)
        temp_mask[0:filled_area.shape[0], 0:filled_area.shape[1]] = filled_area
        
        # Find boundary pixels
        for x in range(temp_mask.shape[0]):
            for y in range(temp_mask.shape[1]):
                if temp_mask[x, y]:
                    # Check boundary condition
                    if (x == 0 or x == temp_mask.shape[0]-1 or 
                        y == 0 or y == temp_mask.shape[1]-1 or
                        not temp_mask[x-1, y] or not temp_mask[x+1, y] or
                        not temp_mask[x, y-1] or not temp_mask[x, y+1]):
                        border_mask[x + minr, y + minc] = [255, 255, 255]
    
    
    # Superpose borders
    border_superposed = cv2.addWeighted(original_image, 0.8, border_mask, 1, 0)
    
    # Print analysis
    print(f'Number of sunspots: {len(regionprops)}')
    print('')
    for i, prop in enumerate(regionprops):
        print(f'Sunspot {i + 1}: {int(prop.area * 1024/original_image.shape[0])} pixels') #RENORMALIZED AREA
    
    # Save results
    img_name = results['image_name']
    output_path = f'{output_folder}/output_{img_name}.jpg'
    cv2.imwrite(output_path, cv2.cvtColor(border_superposed, cv2.COLOR_RGB2BGR))
    print(f'Image with borders saved at {output_path}')


def main():
    #Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_folder', type=str, default='Method2/results/')
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--min_area', type=int, default=3)

    args = parser.parse_args()

    input_path, output_folder, threshold, min_area \
        = args.input_path, args.output_folder, args.threshold, args.min_area
    
    image_name = input_path.split("\\")[-1] 
    image_name = image_name.split('/')[-1]
    image_name = image_name.split('.')[0]

    # Load model
    model = SunspotSegmenter().to(device)
    model.load_state_dict(torch.load('Method2/new_model.pth'))
    
    # Detect sunspots
    results = detect_sunspots(model, input_path, device, threshold)
    
    # Analyze sunspots
    analyze_sunspots(results, min_area, output_folder)


if __name__ == '__main__':
    main()
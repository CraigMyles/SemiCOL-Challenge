import os
import re
import cv2
import numpy as np
import h5py
from tqdm import tqdm
import argparse

class Region:
    def __init__(self, x, y, w, h, d, file_name=None, path=None, case=None):
        #CHANGE d=2.16945 to d=1.90637 if you want to use the LMU cases
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.d = d
        self.file_name = file_name
        self.path = path
        self.case = case

        # Normalized coordinates
        self.x_norm = int(round(int(x) / float(d)))
        self.y_norm = int(round(int(y) / float(d)))
        self.w_norm = int(round(int(w) / float(d)))
        self.h_norm = int(round(int(h) / float(d)))

    def intersects(self, x, y, w, h):
        """
        Returns a list of regions that intersect with the given rectangle (x, y, w, h).
        """
        x1, y1, x2, y2 = self.x_norm, self.y_norm, (self.x_norm + self.w_norm), (self.y_norm - self.h_norm)
        
        #normalise the given rectangle
        x = int(round(int(x) / float(self.d)))
        y = int(round(int(y) / float(self.d)))
        w = int(round(int(w) / float(self.d)))
        h = int(round(int(h) / float(self.d)))
        
        x3, y3, x4, y4 = x, y, (x + w), (y - h)

        # if rectangle has area 0, no intersection
        if x1 == x2 or y1 == y2 or x4 == x3 or y3 == y4:
            return [] # no intersection



        # if x2 < x3 or x4 < x1 or y2 < y3 or y4 < y1:
        #     return []  # no intersection


        
        # If one rectangle is on left side of other
        if x1 > x4 or x3 > x2:
            return []
    
        # If one rectangle is above other
        if y2 > y3 or y4 > y1:
            return []


        # Compute regions that intersect with the given rectangle
        regions = []
        for region in all_regions:
            x1, y1, x2, y2 = region.x, region.y, region.x + region.w, region.y + region.h

            if x2 < x3 or x4 < x1 or y2 < y3 or y4 < y1:
                continue  # no intersection

            # Add region to list of intersecting regions
            regions.append(region)

        return regions
    
    def normalise_to_d(self):
        """
        Returns a region with the same coordinates but normalised to the given d.
        """
        d= self.d
        return Region(int(round(self.x / float(d))), int(round(self.y / float(d))), int(round(self.w / float(d))), int(round(self.h / float(d))), d=d)


all_regions = []


###


def get_args():
    parser = argparse.ArgumentParser(description='Train the ViT on images and target masks')
    parser.add_argument('--manual_path', type=str, help='Path to the 01_manual folder', default='/home/semicol/semicol/01_MANUAL/')
    parser.add_argument('--output', type=str, help='Path to the output .h5 file', default='/home/semicol/semicol/output/semicol.h5')

    return parser.parse_args()

args = get_args()



###


def compare_intersect(query_region, all_regions):
  # for each region in all_regions, check if it intersects with query_region
  # if it does, add it to a list of intersecting regions
  # return the list of intersecting regions
  intersecting_regions = []
  for region in all_regions:
    if region.intersects(query_region.x, query_region.y, query_region.w, query_region.h):
      intersecting_regions.append(region)
  return intersecting_regions


###


# def store_to_h5py(image, ground_truth, metadata, file_name="/data2/semicol/semicol.h5"):
def store_to_h5py(image, ground_truth, metadata, file_name=args.output):
    # print("Storing to h5py file...")
    # open the h5py file in append mode or create it if it doesn't exist
    with h5py.File(file_name, "a") as f:
        # print("Current keys: ", list(f.keys()))
        # create datasets for images and ground truth images, and for metadata
        # if the datasets don't exist already
        if "images" not in f:
            dset_images = f.create_dataset("images", (0, 256, 256, 3), dtype=np.uint8, chunks=(1, 256, 256, 3), maxshape=(None, 256, 256, 3))
            dset_ground_truths = f.create_dataset("ground_truths", (0, 256, 256, 3), dtype=np.uint8, chunks=(1, 256, 256, 3), maxshape=(None, 256, 256, 3))
            dset_metadata = f.create_dataset("metadata", (0,), dtype=h5py.special_dtype(vlen=str), maxshape=(None,))
        else:
            # get references to the existing datasets
            dset_images = f["images"]
            dset_ground_truths = f["ground_truths"]
            dset_metadata = f["metadata"]

        # get the current size of the datasets
        current_size = dset_images.shape[0]

        # append the metadata with the case_id
        case_id = "hi"
        metadata_with_case_id = f"{metadata}, case_id={case_id}"

        # extend the datasets with the new data
        dset_images.resize((current_size + 1, 256, 256, 3))
        dset_ground_truths.resize((current_size + 1, 256, 256, 3))
        dset_metadata.resize((current_size + 1,))
        dset_images[current_size] = image
        dset_ground_truths[current_size] = ground_truth
        dset_metadata[current_size] = metadata_with_case_id


###


# root_directory = "/data2/semicol/DATASET_TRAIN/01_MANUAL/"
root_directory = args.manual_path


# Regex pattern to extract x, y, w, h, and title values
pattern = r"\[d=(\d+\.\d+),x=(\d+),y=(\d+),w=(\d+),h=(\d+)\]"

all_regions = []

for path, subdirs, files in os.walk(root_directory):
	#print(path)  
	dirname = path.split(os.path.sep)[-1]
	if dirname == 'image':   #Find all 'image' directories
		images = os.listdir(path)  #List of all image names in this subdirectory
		for i, image_name in enumerate(images):  
			if image_name.endswith(".png"):   #Only read png images...
				image_path = path+"/"+image_name
				# print(image_path)
				# Extract values using regex
				match = re.search(pattern, image_path)
				if match:
					d, x, y, w, h = map(float, match.groups())
					file_name = image_path.split("/")[-1]
					case = file_name.split()[0]

					# Create Region instance
					region = Region(x, y, w, h, d, file_name, path, case)
					print(region.x_norm, region.y_norm, region.w_norm, region.h_norm)
					#   all_regions = [region]
					all_regions.append(region)
				else:
					# all_regions = []
					print("No match found.")


###


PATCH_DIM = 256
OVERLAP = 0


###



# def add_padding(roi):
#   # Define the desired output shape
#   output_shape = (256, 256, 3)

#   # Calculate the shape of the input image
#   input_shape = roi.shape
  
  

#   # Create an empty array of the desired output shape
#   output_array = np.zeros(output_shape, dtype=roi.dtype)

#   # Calculate the dimensions of the ROI within the output array
#   roi_start = (output_shape[0] - input_shape[0]) // 2
#   roi_end = roi_start + input_shape[0]

#   # Copy the ROI into the output array
#   output_array[roi_start:roi_end, :, :] = roi

#   # Set roi to the padded output array
#   roi = output_array

#   print("Converted from {} to {}.".format(input_shape, roi.shape))

#   return roi

#   # cv2.imwrite("./acceptable.png", roi)


# def add_padding(roi):
#     # Define the desired output shape
#     output_shape = (256, 256, 3)

#     # Calculate the shape of the input image
#     input_shape = roi.shape

#     # Calculate the amount of padding needed to achieve the desired output shape
#     pad_width = [(max(0, (d - o)) // 2, max(0, (d - o)) - max(0, (d - o)) // 2) for d, o in zip(output_shape, input_shape)]
#     pad_width.append((0, 0))

#     # Pad the input image to achieve the desired output shape
#     padded_roi = np.pad(roi, pad_width, mode='constant')

#     cv2.imwrite("./acceptable.png", roi)
#     return padded_roi

def add_padding(roi):
    # Define the desired output shape
    output_shape = (256, 256, 3)

    # Calculate the shape of the input image
    input_shape = roi.shape

    # Calculate the amount of padding needed to achieve the desired output shape
    pad_height = max(output_shape[0] - input_shape[0], 0)
    pad_width = max(output_shape[1] - input_shape[1], 0)
    top, bottom = pad_height // 2, pad_height - (pad_height // 2)
    left, right = pad_width // 2, pad_width - (pad_width // 2)

    # Pad the input image to achieve the desired output shape
    padded_roi = cv2.copyMakeBorder(roi, top, bottom, left, right, cv2.BORDER_CONSTANT)

    # cv2.imwrite("./acceptable.png", padded_roi)
    return padded_roi


###


def single_section_extract(query_region, intersecting_region):
    # print("Single Intersectoin Section Extract")
    # Load region and extract patch.
    intersecting_region = intersecting_region[0]

    # Load the image
    image = cv2.imread(intersecting_region.path+"/"+intersecting_region.file_name)

    roi_x = int(query_region.x_norm - intersecting_region.x_norm)
    roi_y = int(intersecting_region.y_norm - query_region.y_norm)

    roi_w, roi_h = query_region.w_norm, query_region.h_norm


    # Crop the image to the ROI
    roi = np.zeros((roi_h, roi_w, 3), dtype=np.uint8)
    roi = image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    # print("Single Image ROI: "+str(roi.shape))
    image = None #release memory

    # # if roi shape is (256, 193, 3), write that image to disk
    # if roi.shape == (256, 193, 3):
    #     cv2.imwrite("./error.png", roi)
    #     print("Wrote image to disk: "+"./"+str(query_region.case)+"/"+str(query_region.file_name)+"/"+str(intersecting_region.file_name))
    # else:
    #     print("/data2/semicol/patches/"+str(query_region.case)+"/"+str(query_region.file_name)+"/"+str(intersecting_region.file_name))

    # if roi.shape[0] and roi.shape[1] are not 0, pad the image 


    if (roi.shape[0] == 0) or (roi.shape[1] == 0):
        # print("avoided a mask with 0 size")
        return None
    
    if 0 < roi.shape[0] < 256 or 0 < roi.shape[1] < 256:
        # print("adding padding to mask...")
        roi = add_padding(roi)
        # print("added padding to mask. mask size: ", roi.shape)



    #Only write image if the patch is not empty
    if roi.size != 0:
        #retrieve equivalent mask
        mask = cv2.imread(intersecting_region.path.replace("image", "mask") +"/"+intersecting_region.file_name.replace(".png", "-labelled.png"))
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB) # Perhaps dont need conversion if storing array.
        mask = mask[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

        if (mask.shape[0] == 0) or (mask.shape[1] == 0):
            # print("avoided a mask with 0 size")
            return None

        if 0 < mask.shape[0] < 256 or 0 < mask.shape[1] < 256:
            # print("adding padding to mask...")
            mask = add_padding(mask)
            # print("added padding to mask. mask size: ", str(mask.shape))

        source_image_path = (intersecting_region.path+"/"+intersecting_region.file_name)
        source_mask_path = (intersecting_region.path.replace("image", "mask")+"/"+intersecting_region.file_name.replace(".png", "-labelled.png"))

        metadata = {"source_image_path(s)": source_image_path, 
                    "source_mask_path(s)": source_mask_path, 
                    "case": intersecting_region.case, 
                    "query_region": query_region, 
                    "intersecting_region(s)": intersecting_region,
                    "number_of_intersections": int(1), 
                    "patch_dim": PATCH_DIM, 
                    "overlap": OVERLAP, 
                    "patch_type": "single"
                    }
        #file name is last part of args output
        # file_name = args.output.split("/")[-1]


        store_to_h5py(image=roi, ground_truth=mask, metadata=metadata, file_name=args.output)

        # cv2.imwrite(abs_path, roi)
    else:
        print("PATCH IS EMPTY (SINGLE)")
        # pass
        
    
def multi_region_extract(query_region, intersections):
    # print("multi region extract")
    # Create an array filled with zeroes of shape (6000, 6000)
    arr = np.zeros((6000, 6000, 3), dtype=np.uint8)

    # Load the three images and their corresponding coordinates
    images = []
    coordinates = []
    relevant_files = ""
    for region in intersections:
        img = cv2.imread(region.path+"/"+region.file_name)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.imread(filename)
        relevant_files = relevant_files+"_AND_"+region.file_name
    
        images.append(img)
        coordinates.append((region.x_norm, region.y_norm, region.w_norm, region.h_norm))

    # Normalize the coordinates so they fit within the array
    min_x = min(coord[0] for coord in coordinates)
    min_y = min(coord[1] for coord in coordinates)
    max_y = max(coord[1] for coord in coordinates)
    normalized_coordinates = [(coord[0] - min_x, coord[1] - min_y, coord[2], coord[3]) for coord in coordinates]

    # Determine which quadrant each image belongs to and place it in the larger array
    for i in range(len(intersections)):
        x, y, w, h = normalized_coordinates[i]
        # print(f"Image {i} has dimensions {w}x{h} and position ({x}, {y}).")
        if x + w <= arr.shape[1]//2 and y + h <= arr.shape[0]//2:
            arr[y:y+h, x:x+w] = images[i]
        elif x >= arr.shape[1]//2 and y + h <= arr.shape[0]//2:
            arr[y:y+h, x:x+w] = images[i]
        elif x >= arr.shape[1]//2 and y >= arr.shape[0]//2:
            arr[y:y+h, x:x+w] = images[i]
        elif x + w <= arr.shape[1]//2 and y >= arr.shape[0]//2:
            arr[y:y+h, x:x+w] = images[i]
        else:
            print(f"Error: Image {i} has invalid dimensions or position.")
    
    # Convert query_region's normalised coordinates to absolute coordinates to extract the patch ( with size query_region.w_norm by query_region.h_norm from arr
    roi_x = int(query_region.x_norm - min_x)
    # roi_y = int(query_region.y_norm - min_y)
    roi_y = int(max_y - query_region.y_norm)


    roi_w, roi_h = query_region.w_norm, query_region.h_norm  # The width and height of the ROI
    

    # Crop the image to the ROI
    roi = np.zeros((roi_h, roi_w, 3), dtype=np.uint8)
    # print("roi size: ", roi.size)
    roi = arr[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

    # print("current mask shape: ", roi.shape)

    if (roi.shape[0] == 0) or (roi.shape[1] == 0):
        # print("avoided a mask with 0 size")
        return None
    
    if 0 < roi.shape[0] < 256 or 0 < roi.shape[1] < 256:
        # print("adding padding to mask...")
        roi = add_padding(roi)
        # print("added padding to mask. mask size: ", roi.shape)

    #Only write image if the patch is not empty
    if roi.size != 0:
        # Do the same but with the mask
        # Create an array filled with zeroes of shape (6000, 6000)
        arr = np.zeros((6000, 6000, 3), dtype=np.uint8)

        # Load the three images and their corresponding coordinates
        images = []
        coordinates = []
        # temp_file_name = ""
        for region in intersections:
            mask = cv2.imread(region.path.replace("image", "mask")+"/"+region.file_name.replace(".png", "-labelled.png"))
            # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
            images.append(mask)

        # Determine which quadrant each image belongs to and place it in the larger array
        for i in range(len(intersections)):
            x, y, w, h = normalized_coordinates[i]
            # print(f"Image {i} has dimensions {w}x{h} and position ({x}, {y}).")
            if x + w <= arr.shape[1]//2 and y + h <= arr.shape[0]//2:
                arr[y:y+h, x:x+w] = images[i]
            elif x >= arr.shape[1]//2 and y + h <= arr.shape[0]//2:
                arr[y:y+h, x:x+w] = images[i]
            elif x >= arr.shape[1]//2 and y >= arr.shape[0]//2:
                arr[y:y+h, x:x+w] = images[i]
            elif x + w <= arr.shape[1]//2 and y >= arr.shape[0]//2:
                arr[y:y+h, x:x+w] = images[i]
            else:
                print(f"Error: Image {i} has invalid dimensions or position.")
        
        #create a mask variable of size (im_width, im_height, 3) filled with zeroes to store an rgb image
        mask = np.zeros((roi_h, roi_w, 3), dtype=np.uint8)

        mask = arr[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        arr = None #free up memory
        # print("current mask shape: ", mask.shape)

        if (mask.shape[0] == 0) or (mask.shape[1] == 0):
            # print("avoided a mask with 0 size")
            return None
    
        if 0 < mask.shape[0] < 256 or 0 < mask.shape[1] < 256:
            # print("adding padding to mask...")
            mask = add_padding(mask)
            # print("added padding to mask. mask size: ", mask.shape)



        metadata = {"source_image_path(s)": relevant_files, 
                    "source_mask_path(s)": relevant_files.replace(".png", "-labelled.png"),
                    "case": query_region.case, 
                    "query_region": query_region, 
                    "intersecting_region(s)": intersections,
                    "number_of_intersections": len(intersections), 
                    "patch_dim": PATCH_DIM, 
                    "overlap": OVERLAP, 
                    "patch_type": "multi"
                    }
        
        store_to_h5py(image=roi, ground_truth=mask, metadata=metadata, file_name=args.output)
            
        # cv2.imwrite(abs_path, roi)
    else:
        # print("PATCH IS EMPTY (MULTI)")
        pass


###


#list unique cases
unique_cases = list(set([region.case for region in all_regions]))
unique_cases.sort()

for case in unique_cases:
    print("Currently working on case: "+case+"...")

    current_case = []
    x_list = []
    w_list = []
    y_list = []
    h_list = []

    for region in all_regions:
        #where region.case == "ukk_case_02"
        if region.case == case:
            current_case.append(region)
            x_list.append(region.x)
            w_list.append(region.w)
            y_list.append(region.y)
            h_list.append(region.h)


            d = region.d
            # print("d = "+str(d))
            patch_dim = int(round(PATCH_DIM * float(d)))
            # print("patch_dim = "+str(patch_dim)) #patch_dim = 488
            # print("PATCH_DIM = "+str(PATCH_DIM)) #PATCH_DIM = 256
            


    for x in tqdm(range(0, int(max(x_list)+max(w_list)) - patch_dim, patch_dim - OVERLAP), disable=False):
        for y in range(0, int(max(y_list)+max(w_list)) - patch_dim, patch_dim - OVERLAP):
            # print(x,y)

            # Example query for intersecting regions
            query_region = Region(x, y, d=d, w=patch_dim, h=patch_dim, case=case)

            intersections = compare_intersect(query_region, current_case)

            if (intersections):
                # print("Found "+str(len(intersections))+" intersections for query region: "+query_region.file_name)
                # print("intersectoins! :" + str(len(intersections)))
                # get number of intersections
                count = len(intersections)

                if count == 1:
                    # print("ONE WAY INTERSECTION:")
                    # print(intersecting_region.file_name)
                    single_section_extract(query_region, intersections)
                elif 5 > count > 1:
                    # print("MULTI WAY INTERSECTION:")
                    # if count == 3:
                        # print("THREE WAY INTERSECTION:")
                    multi_region_extract(query_region, intersections)
                else:
                    print("ERROR: INVALID NUMBER OF INTERSECTIONS")

                # elif count == 2:
                #     print("TWO WAY INTERSECTION:")

                #     # for intersecting_region in intersections:
                #         # print(intersecting_region.file_name)

                #     multi_region_extract(query_region, intersections)

                # elif count == 3:
                #     print("THREE WAY INTERSECTION:")

                #     # for intersecting_region in intersections:
                #     #     print(intersecting_region.file_name)

                #     multi_region_extract(query_region, intersections)

                # elif count == 4:
                #     print("FOUR WAY INTERSECTION:")

                #     # for intersecting_region in intersections:
                #     #     print(intersecting_region.file_name)

                #     multi_region_extract(query_region, intersections)




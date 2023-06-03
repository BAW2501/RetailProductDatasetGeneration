from pathlib import Path
import cv2 as cv
import numpy as np
from joblib import Parallel, delayed

REMOVE_BACKGROUND = False



data_folder = Path("images")
bounding_box_id_folder = Path("labels")
index_to_class = list(Path("classes.txt").read_text().split("\n"))
output_folder = Path("cropped_images")
# create subfolders for each class
for class_name in index_to_class:
    Path(output_folder / class_name).mkdir(parents=True,exist_ok=True)
    
# Define the background removal function using GrabCut
def remove_background(img):
    mask = np.zeros(img.shape[:2], np.uint8)  # Create a mask with the same shape as the image
    bgd_model = np.zeros((1, 65), np.float64)  # Background model
    fgd_model = np.zeros((1, 65), np.float64)  # Foreground model
    rect = (1, 1, img.shape[1] - 1, img.shape[0] - 1)  # Rectangle enclosing the foreground object

    # Run GrabCut algorithm to extract foreground
    cv.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv.GC_INIT_WITH_RECT)

    # Create a mask where 0 and 2 denote the background, while 1 and 3 denote the foreground
    mask = np.where((mask == 0) | (mask == 2), 0, 1).astype('uint8')

    # Apply the mask to the image to remove the background
    img = img * mask[:, :, np.newaxis]

    return img

def preprocess(image, id_bb):
    assert image.stem == id_bb.stem
    # Read image
    img = cv.imread(str(image), cv.IMREAD_COLOR)
    dh, dw, _ = img.shape
    # Read bounding box (in yolov5 format)
    # Crop image and display it
    id_bb = Path(id_bb).read_text().split()
    class_id = int(id_bb[0])
    class_name = index_to_class[class_id]
    x, y, w, h = [float(i) for i in id_bb[1:]]
    l = int((x - w / 2) * dw)
    r = int((x + w / 2) * dw)
    t = int((y - h / 2) * dh)
    b = int((y + h / 2) * dh)
    l = max(l, 0)
    r = min(r, dw - 1)
    t = max(t, 0)
    b = min(b, dh - 1)
    crop_img = img[t:b, l:r]

    # Remove the background using GrabCut
    if REMOVE_BACKGROUND:
        crop_img = remove_background(crop_img)

    # Save to output folder
    cv.imwrite(str(output_folder / class_name /image.name), crop_img)


image_iter = list(data_folder.iterdir())
id_bbox_iter = list(bounding_box_id_folder.iterdir())

if __name__ == "__main__":
    print("Starting preprocessing")
    print("Number of images:", len(image_iter))
    print("Number of bounding boxes:", len(id_bbox_iter))
    # for image, id_bb in tqdm(zip(image_iter, id_bbox_iter, strict=True)):
    #     preprocess(image, id_bb)
    # parallel processing
    zip_iter = zip(image_iter, id_bbox_iter, strict=True)
    Parallel(n_jobs=-1,verbose=13)(delayed(preprocess)(image, id_bb) for image, id_bb in zip_iter)
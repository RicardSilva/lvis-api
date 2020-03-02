from lvis import LVIS

import argparse
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import cv2
import os
from lvis.colormap import colormap


"""
This script copies all images belonging to a list of classes and assembles the respective 
semantic mask from the existing instance masks
     
"""


def get_img_path(img_dir, img_id, dataset):
    img = dataset.load_imgs([img_id])[0]
    img_path = os.path.join(img_dir, img["file_name"])
    if not os.path.exists(img_path):
        dataset.download(img_dir, img_ids=[img_id])

    return img_path

def coco_segm_to_poly(_list):
    x = _list[0::2]
    y = _list[1::2]
    points = np.asarray([x, y])
    return np.transpose(points)



def buildSemanticMask(image_path, annotations, colors, dest):
    
    tmp = cv2.imread(image_path)
    b, g, r = cv2.split(tmp)
    img = cv2.merge([r, g, b])

    fig = plt.figure(frameon=False)
    fig.set_size_inches(img.shape[1] / 100, img.shape[0] / 100)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_title("")
    ax.axis("off")
    fig.add_axes(ax)

    background = np.zeros(img.shape, np.float32)
    ax.imshow(background)

    for ann in annotations:
        for seg in ann["segmentation"]:
            segm = coco_segm_to_poly(seg)
            color = colors[ann["category_id"] % len(colors), 0:3]
        
            # segm is numpy array of shape Nx2
            polygon = Polygon(
                segm, fill=True, facecolor=color, edgecolor=color, linewidth=0, alpha=1, aa = False
            )
            ax.add_patch(polygon)

    plt.savefig(dest)
    plt.close()

def main():


    # Parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_img', type=str, required=True, help="Source folder of the images, will be created if it doesn't exist already")
    parser.add_argument('--src_ann', type=str, required=True, help='Source folder of the annotations')
    parser.add_argument('--dest', type=str, required=True, help='Destination folder')
    parser.add_argument('--classes', type=str, nargs='+', required=True, help='One or more classes')
    parser.add_argument('--solo', action='store_true', help='Mask only specified classes')
    args = parser.parse_args()

    class_colors = colormap(rgb=True) / 255

    """ Iterate over train, validation subsets """
    for data_type in ["train", "val"]:

        img_dir = os.path.join(args.src_img,'{}2017'.format(data_type))
        ann_file='{}/lvis_v0.5_{}.json'.format(args.src_ann,data_type)

        # initialize LVIS api for instance annotations
        lvis=LVIS(ann_file)


        """ get category ids for specified categories """
        cat_ids = lvis.get_cat_ids(cat_names=args.classes)

        # get img ids that have at least one object of a relevant category
        img_ids = set(lvis.get_img_ids(cat_ids=cat_ids))

        print('Found {} images'.format(len(img_ids)))

        for img_id in img_ids:

        #img_id = list(img_ids)[0]

            # get image path
            img_path = get_img_path(img_dir, img_id, lvis)
                

            """ if not solo get all existing masks for the image """
            if not args.solo:
                cat_ids = None

            ann_ids = lvis.get_ann_ids(img_ids=[img_id], cat_ids=cat_ids)
            annotations = lvis.load_anns(ids=ann_ids)

            if len(annotations) > 0:
                img = os.path.basename(img_path)
                img = img.split('_')[-1] 
                img_name = img.split('.')[0] 

                """ Copy image """
                source = img_path
                destination = os.path.join(args.dest, data_type, img_name + ".jpg")
                try:
                    shutil.copy(source, destination)
                except IOError as io_err:
                    os.makedirs(os.path.dirname(destination))
                    shutil.copy(source, destination)

                """ Build semantic mask from instance masks """
                destination = os.path.join(args.dest, data_type, img_name + "_l.png")       

                buildSemanticMask(img_path, annotations, class_colors, destination)




if __name__ == "__main__":
    main()


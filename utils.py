import matplotlib.pyplot as plt
import os
import cv2
import matplotlib.image as mpimg
import shutil
import xmltodict
import numpy as np 

def get_mask_seg_ellipse(path_to_img):
    # get the image

    img = mpimg.imread(path_to_img)
    basename = os.path.basename(path_to_img)

    # filename_index, e.g. filename = 1.png
    # filename_index = 1, for extracting coordinates
    filename_index = int(os.path.splitext(basename)[0]) - 1
    # print(filename_index)

    
    path_to_coordinates = path_to_img.replace(basename, 'Label')
    #print(path_to_coordinates)
    path_to_coordinates = os.path.join(path_to_coordinates,'Labels.txt')
    #print(path_to_coordinates)
    coordinates,ret = load_coordinates(path_to_coordinates)
    
    
    
    if(coordinates[filename_index]['is_defect']==1):
        mask=mpimg.imread(path_to_coordinates.replace('Labels.txt', coordinates[filename_index]['coord']))
    else:
        mask=np.zeros_like(img)
    # print(coordinates[filename_index]['angle'])

    return mask

def get_coordinates(path_to_label, xml):

    with open(path_to_label, encoding='utf-8') as f:
        if xml:
            label_xml = xmltodict.parse(f.read())

            # print(type(label_xml))
            # print(label_xml)

            coordinates_object = label_xml['annotation']['object']
        else:
            label_txt = f.read()
            coordinates_object = label_txt.strip().split('\n')

    return coordinates_object

def load_coordinates(path_to_coor):

    coord_dict = {}
    coord_dict_all = {}
    defect_pic=[]
    with open(path_to_coor) as f:
        coordinates = f.read().split('\n')
        for coord in coordinates:
            # print(len(coord.split('\t')))
            if len(coord.split('\t')) == 5:
                coord_dict = {}
                coord_split = coord.split('\t')
                # print(coord_split)
                # print('\n')
                coord_dict['is_defect'] = round(float(coord_split[1]))
                coord_dict['name'] = coord_split[2]
                coord_dict['unknown'] = float(coord_split[3])
                coord_dict['coord'] = coord_split[4]
                index = int(coord_split[0]) - 1
                defect_pic.append(index)
                coord_dict_all[index] = coord_dict

            if len(coord.split('\t')) == 6:
                coord_dict = {}
                coord_split = coord.split('\t')
                coord_dict['is_defect'] = round(float(coord_split[1]))
                coord_dict['name'] = coord_split[2]
                coord_dict['unknown'] = float(coord_split[3])
                coord_dict['coord'] = coord_split[5]
                index = int(coord_split[0]) - 1
                coord_dict_all[index] = coord_dict

    return coord_dict_all,defect_pic

def plot_ellipse_seg_test(path_to_img):

    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)

    plt.imshow(mpimg.imread(path_to_img), cmap='gray')
    plt.subplot(1, 2, 2)
    mask = get_mask_seg_ellipse(path_to_img)
    plt.imshow(mask, cmap='gray')
    
#plot_ellipse_seg_test(os.path.join(data_dir2, "0576.png"))

IMAGE_CHANNELS = 1

def load_images_masks(path_to_images, img_type, img_format, resize):

    imgflag,pic_index=load_coordinates(os.path.join(path_to_images,"Label","Labels.txt"))
    image_names = [x for x in os.listdir(path_to_images) if x.endswith(img_type)]
    image_names = [x for x in image_names if int(x.split(".")[0])-1 in pic_index]
    #print(x.split(".")[0])

    image_num = len(image_names)
    images_all = np.empty([image_num, resize[0], resize[1], IMAGE_CHANNELS])
    labels_all = np.empty([image_num, resize[0], resize[1], IMAGE_CHANNELS])

    for image_index in range(image_num):
        image_filename = image_names[image_index]
        # print(image_filename)
        # print(image_filename)
        image = mpimg.imread(os.path.join(path_to_images, image_filename), format=img_format)
        mask = get_mask_seg_ellipse(os.path.join(path_to_images, image_filename))

        if resize:
            image = cv2.resize(image, (resize[0], resize[1]))
            mask = cv2.resize(mask, (resize[0], resize[1]))

        images_all[image_index] = np.reshape(image, (resize[0], resize[1], IMAGE_CHANNELS))
        labels_all[image_index] = np.reshape(mask, (resize[0], resize[1], IMAGE_CHANNELS))

    return images_all, labels_all
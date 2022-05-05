from tops.config import instantiate, LazyConfig
from ssd import utils
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import math

def get_config(config_path):
    cfg = LazyConfig.load(config_path)
    cfg.train.batch_size = 1
    return cfg


def get_dataloader(cfg, dataset_to_visualize):
    if dataset_to_visualize == "train":
        # Remove GroundTruthBoxesToAnchors transform
        cfg.data_train.dataset.transform.transforms = cfg.data_train.dataset.transform.transforms[:-1]
        data_loader = instantiate(cfg.data_train.dataloader)
    else:
        cfg.data_val.dataloader.collate_fn = utils.batch_collate
        data_loader = instantiate(cfg.data_val.dataloader)

    return data_loader


def analyze_something(dataloader, cfg):
    box_ratios = []
    label_dict = dict.fromkeys(range(9), 0)
    for batch in tqdm(dataloader):
        print("batch['labels].shape: ", batch["labels"].shape)
        print("batch['labels']: ", batch["labels"])
        for labels in batch["labels"]:
            print("labels: ", labels)
            for label in range(len(labels)):
                print(labels[label])
                label_dict[int(labels[label])] += 1
        
        #Box ratios:
        for box in batch["boxes"]:
            for shape in box:
                ratio = (shape[2]-shape[0])*1024/((shape[3]-shape[1])*128) # ratio = 1:bredde/høyde
                box_ratios.append(float(ratio))
    
    print("Total boxes: ", len(box_ratios))
    print("Total labels: ", label_dict)
    
    mean = sum(box_ratios) / len(box_ratios)
    var = sum((l-mean)**2 for l in box_ratios) / len(box_ratios)
    st_dev = math.sqrt(var)

    print("Mean of the box ratios is : 1:" + str(mean))
    print("Standard deviation of the box ratios (1:x) is : " + str(st_dev))

    ### Histogram for showing ratios
    # n, bins, patches = plt.hist(box_ratios, 100)

    # plt.xlabel('Ratio 1:x (Height:Width)')
    # plt.ylabel('Intensity')
    # plt.title('Histogram of bounding box ratios in dataset')
    # plt.xlim(0, 2)
    # plt.ylim(0, 4000)
    # plt.grid(True)
    
    # plt.savefig('hist_ratios.png')

    # ratio_05_height = []
    # ratio_1_height = []
    # ratio_15_height = []
    # ratio_else_height = []
    # for batch in tqdm(dataloader):
    #     # Remove the two lines below and start analyzing 😃
    #     for box in batch["boxes"]:
    #         for shape in box:
    #             ratio = (shape[2]-shape[0])*1024/((shape[3]-shape[1])*128) # ratio = 1:bredde/høyde
    #             height = (shape[3]-shape[1])*128
    #             if ratio < 0.5:
    #                 ratio_05_height.append(float(height))
    #             elif ratio < 1:
    #                 ratio_1_height.append(float(height))
    #             elif ratio < 1.5:
    #                 ratio_15_height.append(float(height))
    #             else:
    #                 ratio_else_height.append(float(height))

    # ---------------- Histogram for ratio_05
    # n, bins, patches = plt.hist(ratio_05_height, 100)
    
    # plt.xlabel('Size (height in px)')
    # plt.ylabel('Intensity')
    # plt.title('Histogram of bounding box sizes with ratios between inf:1 to 2:1')
    # plt.xlim(0, 100)
    # plt.ylim(0, 400)
    # plt.grid(True)
    
    # plt.savefig('hist_05_ratios.png')
    
    # ---------------- Histogram for ratio_1
    # n, bins, patches = plt.hist(ratio_1_height, 100)

    # plt.xlabel('Size (height in px)')
    # plt.ylabel('Intensity')
    # plt.title('Histogram of bounding box sizes with ratios between 2:1 and 1:1')
    # plt.xlim(0, 130)
    # plt.ylim(0, 400)
    # plt.grid(True)

    # plt.savefig('hist_1_ratios.png')
    
    # ---------------- Histogram for ratio_1.5
    # n, bins, patches = plt.hist(ratio_15_height, 100)

    # plt.xlabel('Size (height in px)')
    # plt.ylabel('Intensity')
    # plt.title('Histogram of bounding box sizes with ratios between 1:1 and 2:3')
    # plt.xlim(0, 130)
    # plt.ylim(0, 400)
    # plt.grid(True)

    # plt.savefig('hist_1_5_ratios.png')

    # ---------------- Histogram for ratio_else
    # n, bins, patches = plt.hist(ratio_else_height, 100)

    # plt.xlabel('Size (height in px)')
    # plt.ylabel('Intensity')
    # plt.title('Histogram of bounding box sizes with ratios between 2:3 and 1:inf')
    # plt.xlim(0, 130)
    # plt.ylim(0, 400)
    # plt.grid(True)

    # plt.savefig('hist_other_ratios.png')
    
    n, bins, patches = plt.hist(box_ratios, 150)

    plt.xlabel('Size (square root of bounding box area) [px]')
    plt.ylabel('Intensity')
    plt.title('Histogram of bounding box sizes in dataset')
    plt.xlim(0, 300)
    plt.ylim(0, 4000)
    plt.grid(True)
    
    plt.savefig('hist_sizes.png')


def main():
    config_path = "configs/tdt4265.py"
    cfg = get_config(config_path)
    dataset_to_analyze = "train"  # or "val"

    print("Label map is:", cfg.label_map)

    dataloader = get_dataloader(cfg, dataset_to_analyze)
    analyze_something(dataloader, cfg)


if __name__ == '_main_':
    main()
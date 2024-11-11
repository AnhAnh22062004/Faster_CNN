import torch
from torchvision.datasets import VOCDetection
from torchvision.transforms import ToTensor
from pprint import pprint

class VOCDataset(VOCDetection):
    def __init__(self, root, year, image_set, download, transform):
        super().__init__(root, year, image_set, download=download, transform=transform)
        self.categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                           'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                           'train', 'tvmonitor']
        
    def __len__(self):
        return super().__len__()
    
    def __getitem__(self, item):
        image, data = super().__getitem__(item)
        all_boxes = []
        all_labels = []
        for obj in data['annotation']['object']:
            x_min = int(obj["bndbox"]["x_min"])
            y_min = int(obj["bndbox"]["y_min"])
            x_max = int(obj["bndbox"]["x_max"])
            y_max = int(obj["bndbox"]["y_max"])
            
            all_boxes.append([x_min, y_min, x_max, y_max])
            all_labels.append(self.categories.index(obj["name"]))
        
        all_boxes = torch.FloatTensor(all_boxes)
        all_labels = torch.int64(all_labels)
        target = {
            "boxes" : all_boxes,
            "labels" : all_labels
        }
        
        return image, target
        
            
if __name__ == "__main__":
    transform = ToTensor()
    dataset = VOCDataset(root="my_dataset", year="2012", image_set="train", download=True, transform=transform)
    image, target = dataset[2000]
    # print(image.shape)
    # pprint(target)

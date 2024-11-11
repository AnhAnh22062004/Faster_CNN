import os
import numpy as np
import argparse
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pprint import pprint
import cv2

def get_args():
    parser = argparse.ArgumentParser(description="Train faster rcnn model")
    parser.add_argument("--path_image", "-p", default="...", help= "enter path image_test")
    parser.add_argument("--checkpoint_folder", "-c", type=str, default="trained_models",
                        help="Path to save checkpoint")
    parser.add_argument("--saved_checkpoint", "-o", type=str, default=None, help="Continue from this checkpoint")
    parser.add_argument("--conf_threshold", "-c", type = float,default= 0.25, help= "enter conf_threshold")
    args = parser.parse_args()
    return args


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = fasterrcnn_mobilenet_v3_large_320_fpn()
    in_channels = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_channels, num_classes=21)
    checkpoint = torch.load(args.save_checkpoint, map_location= "cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.float()
    #model = model.to(device)
    org_image = cv2.imread(args.path_image)
    image = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB)
    image = np.transpose(image, (2, 0, 1)) / 255
    image = [torch.from_numpy(image).float()]

    model.eval()
    with torch.no_grad():
        output = model(image)[0]
        bboxes = output['bboxes']
        labels = output['labels']
        scores = output['scores']
        for bbox, label, score in zip(bbox, label, score):
            if score > args.conf_threshold:
                x_min, y_min, x_max, y_max = bbox
                cv2.rectangle(org_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 3)
                category = category["label"]
                cv2.putText(org_image, category, (int(x_min), int(y_min)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
                
        cv2.imwrite("prediction.jpg", org_image)


if __name__ == '__main__':
    args = get_args()
    test(args)
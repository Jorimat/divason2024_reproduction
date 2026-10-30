import cv2 as cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import math
from pathlib import Path
import os
import xml.etree.ElementTree as ET


from config import *


def predict_image_batch(images, confidence_threshold, model, device):
    output=[]
    scores = []
    final_boxes = []
    with torch.no_grad():
        preds = model(images)
        for npred, pred_uniq in enumerate(preds):
            cuales = (pred_uniq['scores']>confidence_threshold).detach().cpu().numpy()
            boxes = pred_uniq['boxes'].cpu().numpy().astype(np.int32)
            img = np.ascontiguousarray(images[npred].permute(1,2,0).detach().cpu().numpy(), dtype=np.float32)
            img_cp = img.copy()
            num_varroas = np.sum(cuales)
            aux_boxes=[]
            if num_varroas>0:
                for nbox, box in enumerate(boxes[cuales]):
                    cv2.rectangle(img,(box[0], box[1]),(box[2], box[3]),(255, 0, 0),2)
                    aux_boxes.append((box[0], box[1], box[2], box[3]))
            final_boxes.append(aux_boxes)
            s = pred_uniq['scores'].cpu().numpy()[cuales]
            scores.append(s)
            output.append(img*255)
    return output, final_boxes, scores


def predict_image(input_image_name, model, bbox_params, device):

    model.eval()
    preds_prob = []
    targets_num = []

    valid_imagenes=[]
    positions=[]

    input_image = cv2.imread(input_image_name)
    input_image = input_image.astype(np.float32)
    input_image /= 255.0
    input_image = np.array(input_image)

    pos_x, pos_y = 0, 0

    size_img_y, size_img_x, _ = input_image.shape

    image_orig = input_image
    boxes_orig = []
    labels = torch.ones(0, dtype=torch.int64)
    while True:
        x_max = pos_x+CROP_X
        y_max = pos_y+CROP_Y
        if x_max>=size_img_x:
            x_max = size_img_x
        if y_max>=size_img_y:
            y_max = size_img_y

        transform = A.Compose([A.Crop(x_min = pos_x, y_min=pos_y, x_max=x_max, y_max=y_max),
                               ToTensorV2()],
                               bbox_params=bbox_params)
        result_transform = transform(image=image_orig, bboxes=boxes_orig, labels=labels)
        valid_imagenes.append(result_transform['image'])
        positions.append((pos_x, pos_y))
        pos_x += CROP_X
        if pos_x>=size_img_x:
            pos_x = 0
            pos_y += CROP_Y
            # Sal
            if pos_y >= size_img_y:
                break

    n_batch = 36 # n_batch = len(valid_imagenes) will make things slower, but will always work.
    parte = len(valid_imagenes)//n_batch # This is the batch size
    trozos=[]
    final_boxes = []
    scores = []
    for t in range(n_batch):
        images = list(image.to(device) for image in valid_imagenes[t*parte:(t+1)*parte])
        r, bb, s = predict_image_batch(images, confidence_threshold=CONFIDENCE_THRESHOLD, model=model, device=device)
        final_boxes.extend(bb)
        scores.extend(s)
        trozos+=r
    bboxes = []
    for ((x,y), b) in zip(positions, final_boxes):
        if b == []:
            bboxes.append([])
        else:
            aux_boxes = []

            for (x_min, y_min, x_max, y_max) in b:
                aux_boxes.append((x_min + x , y_min + y,  x_max + x, y_max + y))
            bboxes.append(aux_boxes)

    resultado = []

    nrows = math.ceil(size_img_y / CROP_Y)
    ncols = math.ceil(size_img_x / CROP_X)
    i=0
    for row in range(nrows):
        indices = [i + j for j in range(ncols)]
        imagenes_fila = [trozos[i] for i in indices]
        i=(row+1)*ncols
        imagen_final_fila = cv2.hconcat(imagenes_fila)
        resultado.append(imagen_final_fila)

    resultado = cv2.vconcat(resultado)
    name = Path(input_image_name).stem

    #!mkdir output
    path = "output"
    isExist = os.path.exists(path)
    if not isExist:
      # Create a new directory because it does not exist
      os.makedirs(path)
    #cv2.imwrite("output/"+name+".jpg", resultado)
    return positions, bboxes, scores

    
def predict_small_image(input_image, model, device):
    model.eval()
    transform = A.Compose([ToTensorV2()])
    result_transform = transform(image=input_image)
    images = []
    images.append(result_transform['image'])
    image = list(image.to(device) for image in images)
    with torch.no_grad():
      preds = model(image)
      for npred, pred_uniq in enumerate(preds):
          cuales = (pred_uniq['scores']>0.50).detach().cpu().numpy()
          boxes = pred_uniq['boxes'].cpu().numpy().astype(np.int32)
          scores = pred_uniq['scores'].cpu().numpy()
          return boxes[cuales], scores[cuales]

    
def bounding_boxes_intersection_area (bb1, bb2):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(bb1[0], bb2[0])
    yA = max(bb1[1], bb2[1])
    xB = min(bb1[2], bb2[2])
    yB = min(bb1[3], bb2[3])
    
    # compute the area of intersection rectangle
    interArea = max(xB - xA, 0) * max(yB - yA, 0)
    return interArea


def improve_bboxes(input_image, positions, bboxes, scores, model, device):
  size_img_y, size_img_x, _ = input_image.shape
  bboxes2 = []
  new_boxes = []
  new_scores = []
  for i, (pos, pos_bbs) in enumerate(zip(positions,bboxes)):
    pos_bbs2 = []
    pos_x, pos_y = pos
    x_max = pos_x+CROP_X
    y_max = pos_y+CROP_Y
    if x_max>=size_img_x: # Si nos vamos a pasar en la X:
        x_max = size_img_x
    if y_max>=size_img_y: # Si nos vamos a pasar en la Y:
        y_max = size_img_y
    for j, bb in enumerate(pos_bbs):
      bbx_min, bby_min, bbx_max, bby_max = bb
      if not (((bbx_min <= pos_x + 10) and (bbx_min != 0)) or ((bby_min <= pos_y +10) and (bby_min != 0)) or
          ((bbx_max >= x_max-10) and (bbx_max != size_img_x)) or ((bby_max >= y_max-10) and (bby_max != size_img_y))):
          pos_bbs2.append(bb)
          new_scores.append(scores[i][j])
      else:
          center_x = (bbx_min + bbx_max) // 2
          center_y = (bby_min + bby_max) // 2
          new_x_min = center_x - (CROP_X // 2)
          new_x_max = center_x + (CROP_X // 2)
          new_y_min = center_y - (CROP_Y // 2)
          new_y_max = center_y + (CROP_Y // 2)
          if new_x_min<=0:
            new_x_min = 0
          if new_y_min <= 0:
            new_y_min = 0
          if new_x_max>=size_img_x:
            new_x_max = size_img_x
          if new_y_max>=size_img_y:
            new_y_max = size_img_y
          new_crop = input_image[new_y_min:new_y_max,new_x_min:new_x_max]
          new_boxes_predict, new_scores_predict = predict_small_image(new_crop, model, device)

          for box, new_score in zip(new_boxes_predict, new_scores_predict):
            new_box = (box[0]+new_x_min,box[1]+new_y_min,box[2]+new_x_min,box[3]+new_y_min)
            interArea = bounding_boxes_intersection_area(new_box,bb)
            bbArea = abs((bb[2] - bb[0]) * (bb[3] - bb[1]))
            new_box_add = False
            if (interArea >= bbArea*0.5):
              new_box_add = True
              for bb2 in new_boxes:
                interArea2 = bounding_boxes_intersection_area(bb2,new_box)
                bb2Area = abs((bb2[2] - bb2[0]) * (bb2[3] - bb2[1]))
                if (interArea2 >= 0.5*bb2Area):
                  new_box_add = False

            if new_box_add:
              pos_bbs2.append(new_box)
              new_boxes.append(new_box)
              new_scores.append(new_score)

    bboxes2.append(pos_bbs2)
  return bboxes2, new_scores


def improved_predict_image(input_image_name, model, bbox_params, device):
    positions, bboxes, scores = predict_image(input_image_name, model, bbox_params, device)
    input_image = cv2.imread(input_image_name)
    input_image = input_image.astype(np.float32)
    input_image /= 255.0
    input_image = np.array(input_image)
    bboxes2, scores = improve_bboxes(input_image, positions, bboxes, scores, model, device)
    return bboxes2, scores


def get_true_bboxes(file_lbl):
    tree = ET.parse(file_lbl)
    root = tree.getroot()
    bboxes = [
        [
            int(obj.find("bndbox/xmin").text),
            int(obj.find("bndbox/ymin").text),
            int(obj.find("bndbox/xmax").text),
            int(obj.find("bndbox/ymax").text),
        ]
        for obj in root.findall("object")
    ]
    return bboxes





import re
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import FasterRCNN, fasterrcnn_resnet50_fpn_v2, fasterrcnn_resnet50_fpn, fasterrcnn_mobilenet_v3_large_320_fpn

def prepare_model(file_model, verbose = False):
    bbox_params = A.BboxParams(format = 'pascal_voc',
             min_visibility = 0.6,
             label_fields = ['labels'])
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    if verbose:
        print(device)
    
    # backbone_name = file_model.split('/')[-2].split('_')[1]
    match = re.search(r"resnet[0-9]+", file_model)
    if match:
        backbone_name = match.group()
    if verbose:
        print(backbone_name)
    backbone = resnet_fpn_backbone(
        backbone_name=backbone_name,
        weights=None
    )
    model = FasterRCNN(backbone, num_classes=2, box_nms_thresh=0.1)
    _ = model.to(device)
    
    if device.type == 'gpu' or device.type == 'cuda':
        model.load_state_dict(torch.load(file_model, map_location='cuda:0', weights_only=True))
    elif device.type == 'cpu':
        model.load_state_dict(torch.load(file_model, map_location=torch.device('cpu'), weights_only=True))
    else:
        print("PROBLEM")

    return model, device, bbox_params




import pandas as pd

def predict_val(folder_data, bbox_params, model, device, verbose = False):
    
    # Read data overview file
    df = pd.read_csv(f'{folder_data}/df_dataset.csv')
    
    # List validation image names
    val_names = list(set(df[~df['is_train']]['name']))
    
    # Perform the prediction with the refinement step enabled
    all_bboxes_true, all_bboxes_pred, all_scores = [], [], []
    
    for f in range(len(val_names)):
        if verbose:
            print(val_names[f], end=", ")
    
        # Load true bboxes
        file_lbl = f'{folder_data}/dataset/labels/{val_names[f]}.xml'
        bboxes_true = get_true_bboxes(file_lbl)
        if verbose:
            print(len(bboxes_true), 'labeled', end=", ")
        
        # Predict
        file_img = f'{folder_data}/dataset/images/{val_names[f]}.jpg'
        bboxes_pred, scores = improved_predict_image(file_img, model, bbox_params, device)
        
        # Flatten
        tmp = []
        for bbox_list in bboxes_pred:
            if len(bbox_list) > 0:
                for bbox in bbox_list:
                    tmp.append(bbox)
        bboxes_pred = tmp
        
        # Append
        all_bboxes_true.append(bboxes_true)
        all_bboxes_pred.append(bboxes_pred)
        all_scores.append(scores)

        if verbose:
            print(len(bboxes_pred), 'detected')
    
    return all_bboxes_true, all_bboxes_pred, all_scores




from torchmetrics.detection import MeanAveragePrecision

def evaluate(all_bboxes_true, all_bboxes_pred, all_scores):
    
    metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.50])
    
    preds = []
    targets = []
    
    for bboxes_true, bboxes_pred, scores in zip(all_bboxes_true, all_bboxes_pred, all_scores):
        preds.append({
            "boxes": torch.tensor(bboxes_pred, dtype=torch.float32).reshape(-1, 4),
            "scores": torch.tensor(scores, dtype=torch.float32),
            "labels": torch.ones(len(bboxes_pred), dtype=torch.int64)
        })
    
        targets.append({
            "boxes": torch.tensor(bboxes_true, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.ones(len(bboxes_true), dtype=torch.int64)
        })
    
    metric.update(preds, targets)
    results = metric.compute()
    for key in results.keys():
        results[key] = results[key].item()


    return results
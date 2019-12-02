"""
Process a video (like in `video_demo.py`, but does not use Writers and some other superfluous dependencies)
"""

import os
import time

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch.nn as nn
import torch.utils.data
import numpy as np

from pPose_nms import pose_nms, write_json
from SPPE.src.utils.img import im_to_torch

from yolo.darknet import Darknet
from yolo.preprocess import prep_frame
from yolo.util import dynamic_write_results

from dataloader import crop_from_dets
from dataloader import Mscoco
from SPPE.src.main_fast_inference import InferenNet, InferenNet_fast
from SPPE.src.utils.eval import getMultiPeakPrediction, getPrediction
from matching import candidate_reselect as matching

import cv2

from opt import opt

def main():
    ''' arg parsing '''
    args = vars(opt)
    print(args)

    if not os.path.exists(args['outputpath']):
        os.mkdir(args['outputpath'])
    
    videofile = args['video']
    if not len(videofile):
        raise IOError('Error: must contain --video')

    inp_dim = int(args['inp_dim'])
    confidence = args['confidence']
    num_classes = args['num_classes']
    nms_thresh = args['nms_thesh']

    ''' load input video stream '''
    cap = cv2.VideoCapture(videofile)

    # read_frames = 0
    # while True:
    #     ret, frame = cap.read()
    #     if ret:
    #         cv2.imwrite('frame.jpg', frame)
    #         read_frames += 1
    #     else:
    #         break
    # print("Read %d frames in total" % (read_frames,))

    ''' load detection model '''
    det_model = Darknet("yolo/cfg/yolov3-spp.cfg")
    det_model.load_weights("models/yolo/yolov3-spp.weights")
    det_model.net_info['height'] = inp_dim
    det_model.cuda()
    det_model.eval()
    batch_size = 1

    ''' load pose model '''
    pose_dataset = Mscoco()
    if args['fast_inference']:
        pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
    else:
        pose_model = InferenNet(4 * 1 + 1, pose_dataset)
    pose_model.cuda()
    pose_model.eval()


    ''' iterate over stream '''
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        img, orig_img, dims = prep_frame(frame, inp_dim)
        dims = torch.FloatTensor([dims]).repeat(1, 2)

        with torch.no_grad():
            ''' human detection '''
            img = img.cuda()
            prediction = det_model(img, CUDA=True)

            dets = dynamic_write_results(prediction, confidence, num_classes, nms=True, nms_conf=nms_thresh)

            if isinstance(dets, int) or dets.shape[0] == 0:
                continue

            dets = dets.cpu()

            dims = torch.index_select(dims, 0, dets[:, 0].long())
            scaling_factor = torch.min(inp_dim / dims, 1)[0].view(-1, 1)

            # coordinate transfer
            dets[:, [1, 3]] -= (inp_dim - scaling_factor * dims[:, 0].view(-1, 1)) / 2
            dets[:, [2, 4]] -= (inp_dim - scaling_factor * dims[:, 1].view(-1, 1)) / 2

            
            dets[:, 1:5] /= scaling_factor
            for j in range(dets.shape[0]):
                dets[j, [1, 3]] = torch.clamp(dets[j, [1, 3]], 0.0, dims[j, 0])
                dets[j, [2, 4]] = torch.clamp(dets[j, [2, 4]], 0.0, dims[j, 1])
            boxes = dets[:, 1:5]
            scores = dets[:, 5:6]

            #print(dets[:,0]) # that's the batch index
            
            for box in boxes:
                orig_img = cv2.rectangle(orig_img, tuple(box[:2]), tuple(box[2:]), (255, 255, 255))
            
            cv2.imwrite('frame_%d.jpg'%frame_idx, orig_img)

            if isinstance(boxes, int) or boxes.shape[0] == 0:
                continue

            inps = torch.zeros(boxes.size(0), 3, opt.inputResH, opt.inputResW)
            pt1 = torch.zeros(boxes.size(0), 2)
            pt2 = torch.zeros(boxes.size(0), 2)

            
            # self.Q.put((orig_img[k], im_name[k], boxes, scores[dets[:,0]==k], inps, pt1, pt2))               
            # end of DetectionLoader.update
            # start of DetectionProcessor.update
            # (orig_img, im_name, boxes, scores, inps, pt1, pt2) = self.detectionLoader.read()

            if boxes is None or boxes.nelement() == 0:
                continue

            inp = im_to_torch(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
            inps, pt1, pt2 = crop_from_dets(inp, boxes, inps, pt1, pt2)

            # self.Q.put((inps, orig_img, im_name, boxes, scores, pt1, pt2))
            # end of DetectionProcessor.update
            # beggining of video_demo.py main loop (more specifically line 75)

            ''' pose estimation '''
            hm = pose_model(inps.cuda())
            hm = hm.cpu()#.data()
            #print(frame_idx, hm)

            # end of video_demo.py main loop
            # beggining of DataWriter.save (which is redirected to DataWriter.update)

            if args['matching']:
                preds = getMultiPeakPrediction(
                    hm, pt1.numpy(), pt2.numpy(), args['inputResH'], args['inputResW'], args['outputResH'], args['outputResW'])
                result = matching(boxes, scores.numpy(), preds)
            else:
                preds_hm, preds_img, preds_scores = getPrediction(
                    hm, pt1, pt2, args['inputResH'], args['inputResW'], args['outputResH'], args['outputResW'])
                result = pose_nms(
                    boxes, scores, preds_img, preds_scores)

            print(len(result))

            # TODO: find key points and see if they match `video_demo.py` JSON output (apparently they do not, check how JSON is written)
            for r in result:
                print(frame_idx, r['keypoints'])
                
        frame_idx += 1


        #exit(-1)


main() if __name__ == '__main__' else True
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

from pPose_nms import pose_nms
from SPPE.src.utils.img import im_to_torch

from yolo.darknet import Darknet
from yolo.preprocess import prep_frame
from yolo.util import dynamic_write_results

from dataloader import crop_from_dets
from dataloader import Mscoco
from SPPE.src.main_fast_inference import InferenNet_fast
from SPPE.src.utils.eval import getMultiPeakPrediction, getPrediction
from matching import candidate_reselect as matching

from fn import vis_frame_fast

import cv2

import opt

def dets_from_image(det_model, args, img, dims):
    inp_dim = int(args['inp_dim'])
    confidence = args['confidence']
    num_classes = args['num_classes']
    nms_thresh = args['nms_thesh']

    prediction = det_model(img, CUDA=True)

    dets = dynamic_write_results(prediction, confidence, num_classes, nms=True, nms_conf=nms_thresh)

    if isinstance(dets, int) or dets.shape[0] == 0:
        return None, None

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

    return boxes, scores

def pose_from_dets(pose_model, args, boxes, scores, orig_img):
    inps = torch.zeros(boxes.size(0), 3, args['inputResH'], args['inputResW'])
    pt1 = torch.zeros(boxes.size(0), 2)
    pt2 = torch.zeros(boxes.size(0), 2)

    inp = im_to_torch(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
    inps, pt1, pt2 = crop_from_dets(inp, boxes, inps, pt1, pt2)

    hm = pose_model(inps.cuda())
    hm = hm.cpu()#.data()

    if args['matching']:
        preds = getMultiPeakPrediction(
            hm, pt1.numpy(), pt2.numpy(), args['inputResH'], args['inputResW'], args['outputResH'], args['outputResW'])
        result = matching(boxes, scores.numpy(), preds)
    else:
        preds_hm, preds_img, preds_scores = getPrediction(
            hm, pt1, pt2, args['inputResH'], args['inputResW'], args['outputResH'], args['outputResW'])
        result = pose_nms(
            boxes, scores, preds_img, preds_scores)

    return result

def main():
    ''' arg parsing '''
    args = vars(opt.parser.parse_args())
    print(args)

    output_path = args['outputpath']
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
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
    det_model = Darknet("data/checkpoints/pose/yolo/yolov3-spp.cfg")
    det_model.load_weights("data/checkpoints/pose/yolo/yolov3-spp.weights")
    det_model.net_info['height'] = inp_dim
    det_model.cuda()
    det_model.eval()
    batch_size = 1

    ''' load pose model '''
    pose_dataset = Mscoco()
    if args['fast_inference']:
        pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
    else:
        raise NotImplementedError("Adapted code for InferenNet_fast only")
        #pose_model = InferenNet(4 * 1 + 1, pose_dataset)
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
            t0 = time.time()

            ''' human detection '''
            img = img.cuda()

            boxes, scores = dets_from_image(det_model, args, img, dims)
            if boxes is None or scores is None:
                continue

            #print(dets[:,0]) # that's the batch index
            
            # self.Q.put((orig_img[k], im_name[k], boxes, scores[dets[:,0]==k], inps, pt1, pt2))               
            # end of DetectionLoader.update
            # start of DetectionProcessor.update
            # (orig_img, im_name, boxes, scores, inps, pt1, pt2) = self.detectionLoader.read()

            if boxes is None or boxes.nelement() == 0:
                continue

            # self.Q.put((inps, orig_img, im_name, boxes, scores, pt1, pt2))
            # end of DetectionProcessor.update
            # beggining of video_demo.py main loop (more specifically line 75)

            ''' pose estimation '''
            # end of video_demo.py main loop
            # beggining of DataWriter.save (which is redirected to DataWriter.update)

            result = pose_from_dets(pose_model, args, boxes, scores, orig_img)

            tf = time.time()
            total_time = tf-t0
            print("[%d] Computed %d results in %f seconds (%f FPS)" % (frame_idx, len(result), total_time, (1/float(total_time))))


            ''' output images '''
            cv2.imwrite(os.path.join(args['outputpath'], 'input_%d.jpg'%frame_idx), orig_img)

            dets_img = orig_img.copy()
            for box in boxes:
                dets_img = cv2.rectangle(dets_img, tuple(box[:2]), tuple(box[2:]), (255, 255, 255))
            cv2.imwrite(os.path.join(args['outputpath'], 'dets_%d.jpg'%frame_idx), dets_img)

            frame_with_joints = vis_frame_fast(orig_img, {'imgname': "%d" % frame_idx, 'result': result}, format='coco')
            cv2.imwrite(os.path.join(args['outputpath'], 'joints_%d.jpg'%frame_idx), frame_with_joints)

                
        frame_idx += 1


main() if __name__ == '__main__' else True
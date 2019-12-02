''' the obvious '''
import torch
import numpy as np
import cv2

''' the inherited '''
from dataloader import Mscoco
from dataloader import crop_from_dets

from SPPE.src.main_fast_inference import InferenNet_fast
from SPPE.src.utils.img import im_to_torch
from SPPE.src.utils.eval import getMultiPeakPrediction, getPrediction

from yolo.darknet import Darknet
from yolo.preprocess import prep_frame
from yolo.util import dynamic_write_results

from matching import candidate_reselect as matching

from pPose_nms import pose_nms


class HumanPoseDetector():
    def __init__(self, model_path, weights_path, input_dimension, use_gpu:bool):
        self.use_gpu = use_gpu

        self.detection_model = Darknet(model_path)
        self.detection_model.load_weights(weights_path)
        self.detection_model.net_info['height'] = input_dimension
        self.detection_model.cuda() if self.use_gpu else self.detection_model.cpu()
        self.detection_model.eval()

        pose_dataset = Mscoco()
        self.pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset, use_gpu=self.use_gpu)
        self.pose_model.cuda() if self.use_gpu else self.detection_model.cpu()
        self.pose_model.eval()
        

        ''' arguments from `*demo.py` command line (`from opt import opt`)'''
        self.batch_size = 1
        
        self.inputResH = 320
        self.inputResW = 256
        self.outputResH = 80
        self.outputResW = 64
        
        self.matching:bool = False # not sure about what it is -> test it!
        
        self.nms_thresh = .6
        self.confidence = .05
        self.num_classes = 80 # COCO?

    def detect(self, image):
        def empty_return(): # placeholder
            return None

        print("net info height", self.detection_model.net_info['height'])

        image, original_image, dims = prep_frame(image, self.detection_model.net_info['height'])

        print("dims after prep_frame", dims)
        #exit(-1)

        dims = torch.FloatTensor([dims]).repeat(1, 2)                                                                                           # pylint: disable=no-member

        with torch.no_grad():
            ''' human detection '''
            img = image.cuda() if self.use_gpu else image.cpu()
            prediction = self.detection_model(img, CUDA=self.use_gpu)

            dets = dynamic_write_results(prediction, self.confidence, self.num_classes, nms=True, nms_conf=self.nms_thresh)

            if isinstance(dets, int) or dets.shape[0] == 0:
                return empty_return()

            dets = dets.cpu()

            dims = torch.index_select(dims, 0, dets[:, 0].long())                                                                               # pylint: disable=no-member
            scaling_factor = torch.min(self.detection_model.net_info['height'] / dims, 1)[0].view(-1, 1)                                        # pylint: disable=no-member

            # coordinate transfer
            dets[:, [1, 3]] -= (self.detection_model.net_info['height'] - scaling_factor * dims[:, 0].view(-1, 1)) / 2
            dets[:, [2, 4]] -= (self.detection_model.net_info['height'] - scaling_factor * dims[:, 1].view(-1, 1)) / 2

            
            dets[:, 1:5] /= scaling_factor
            for j in range(dets.shape[0]):
                dets[j, [1, 3]] = torch.clamp(dets[j, [1, 3]], 0.0, dims[j, 0])                                                                 # pylint: disable=no-member
                dets[j, [2, 4]] = torch.clamp(dets[j, [2, 4]], 0.0, dims[j, 1])                                                                 # pylint: disable=no-member
            boxes = dets[:, 1:5]
            scores = dets[:, 5:6]

            #print(dets[:,0]) # that's the batch index
            
            for box in boxes:
                original_image = cv2.rectangle(original_image, tuple(box[:2]), tuple(box[2:]), (255, 255, 255))                                 # pylint: disable=no-member
            
            cv2.imwrite('frame.jpg', original_image)                                                                                            # pylint: disable=no-member

            if isinstance(boxes, int) or boxes.shape[0] == 0:
                return empty_return() 
            
            inps = torch.zeros(boxes.size(0), 3, self.inputResH, self.inputResW)                                                                # pylint: disable=no-member
            pt1 = torch.zeros(boxes.size(0), 2)                                                                                                 # pylint: disable=no-member
            pt2 = torch.zeros(boxes.size(0), 2)                                                                                                 # pylint: disable=no-member

            
            # self.Q.put((orig_img[k], im_name[k], boxes, scores[dets[:,0]==k], inps, pt1, pt2))               
            # end of DetectionLoader.update
            # start of DetectionProcessor.update
            # (orig_img, im_name, boxes, scores, inps, pt1, pt2) = self.detectionLoader.read()

            if boxes is None or boxes.nelement() == 0:
                return empty_return() 

            inp = im_to_torch(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))                                                                        # pylint: disable=no-member
            inps, pt1, pt2 = crop_from_dets(inp, boxes, inps, pt1, pt2)

            # self.Q.put((inps, orig_img, im_name, boxes, scores, pt1, pt2))
            # end of DetectionProcessor.update
            # beggining of video_demo.py main loop (more specifically line 75)

            ''' pose estimation '''
            inps = inps.cuda() if self.use_gpu else inps.cpu()
            hm = self.pose_model(inps)
            hm = hm.cpu()#.data()
            #print(frame_idx, hm)

            # end of video_demo.py main loop
            # beggining of DataWriter.save (which is redirected to DataWriter.update)

            if self.matching:
                preds = getMultiPeakPrediction(
                    hm, pt1.numpy(), pt2.numpy(), self.inputResH, self.inputResW, self.outputResH, self.outputResW)
                result = matching(boxes, scores.numpy(), preds)
            else:
                preds_hm, preds_img, preds_scores = getPrediction(
                    hm, pt1, pt2, self.inputResH, self.inputResW, self.outputResH, self.outputResW)
                result = pose_nms(
                    boxes, scores, preds_img, preds_scores)

            return result

def main():
    ''' hardcoded args '''
    inp_dim = 608
    image_path = "/home/jose/Pictures/umhomemlindo.jpg"
    use_gpu = False

    detector = HumanPoseDetector("yolo/cfg/yolov3-spp.cfg", "models/yolo/yolov3-spp.weights", inp_dim, use_gpu=use_gpu)
    image = cv2.imread(image_path)                                                                                                              # pylint: disable=no-member       

    results = detector.detect(image)

    ''' inspects results '''
    print(results)
    for d in results:
        print(len(d['keypoints']), len(d['kp_score']))

main() if __name__ == '__main__' else True
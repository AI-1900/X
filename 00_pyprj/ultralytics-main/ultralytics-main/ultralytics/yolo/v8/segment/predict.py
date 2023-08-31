# Ultralytics YOLO 🚀, AGPL-3.0 license

import numpy as np
import torch

import sys
sys.path.append("D:/03_vastai/00_task/00_yolov8_nms/ultralytics-main/ultralytics-main/")
# 即 ultralytics文件夹 所在绝对路径


from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.v8.detect.predict import DetectionPredictor

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s : %(levelname)s - [%(filename)s:%(lineno)s] - %(message)s')

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler('log_file.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# 设置打印选项
torch.set_printoptions(precision=4, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=None)

class SegmentationPredictor(DetectionPredictor):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = 'segment'

    def postprocess(self, preds, img, orig_imgs):
        base_path = 'D:\\03_vastai\\00_task\\00_yolov8_nms\\ultralytics-main\\ultralytics-main\\ultralytics\\yolo\\v8\\segment\\v8-seg\\'
        
        """TODO: filter by classes."""
        p = ops.non_max_suppression(preds[0],
                                    self.args.conf,
                                    self.args.iou,
                                    agnostic=self.args.agnostic_nms,
                                    max_det=self.args.max_det,
                                    nc=len(self.model.names),
                                    classes=self.args.classes)
        
        print('\n\n\n[NMS_PARA START]================================')
        print('self.args.conf:', self.args.conf)
        print('self.args.iou:', self.args.iou)
        print('self.args.agnostic_nms:', self.args.agnostic_nms)
        print('self.args.max_det:', self.args.max_det)
        print('nc:', len(self.model.names))
        print('self.args.classes:', self.args.classes)
        print('[NMS_PARA FINISH]================================\n\n\n')

        print('nms res cnt:', len(p))
        
        for cnt in range(0, len(p)):
            path_np_p = base_path + 'loop%d_nms_res_f16.bin'%cnt            
            np_p = p[cnt].numpy().astype(np.float16)
            np_p.tofile(path_np_p)   
            
            print('\ntype(np_p):', type(np_p), 'np_p.dtype:', np_p.dtype, 'np_p.shape:', np_p.shape)
            print('\nnp_p:\n', np_p)
            
        # print('\np:\n', p, '\n')        
        # path_p = base_path + 'nms_res_f16.bin'
        # np_p = p.numpy()
        # print('\nnp_p:\n', np_p, '\n')  
        # np_p.tofile(path_p)
        
        results = []
        proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]  # second output is len 3 if pt, but only 1 if exported
        
        print('\n')
  
        # print('img.shape:', img.shape)
        # print('self.args.retina_masks:', self.args.retina_masks)
  
        # print('\nproto:', proto.shape, '\n')  
        # print('\nproto:', proto, '\n')  
        
        for i, pred in enumerate(p):
            # print('\npred:', pred.shape, '\n')  
            # print('\npred:', pred, '\n')  
        
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs            
            path = self.batch[0]
            img_path = path[i] if isinstance(path, list) else path
            
            # print('orig_img shape:', orig_img.shape)
            # print('path:', path)
            # print('img_path:', img_path)
            
            print('\n[LOOP_%d INFO START]----------------------------------------------------------------\n'%i)
            # print('proto:', proto)
            print('proto:', proto.shape)
            # print('pred:', pred)
            print('pred:', pred.shape)
            # print('orig_img:', orig_img)
            print('orig_img:', orig_img.shape)
            
            print('len(pred):', len(pred))

            np_proto = proto.numpy()
            np_pred = pred.numpy()
            print('np_proto.shape:', np_proto.shape)
            print('np_pred.shape:', np_pred.shape)
          
            path_proto = base_path + 'loop%d_proto_%d_%d_%d_%d.bin'%(i, proto.shape[0], proto.shape[1], proto.shape[2], proto.shape[3])
            path_pred = base_path + 'loop%d_pred_%d_%d.bin'%(i, pred.shape[0], pred.shape[1])
            path_orig_img = base_path + 'loop%d_orig_img_h%d_w%d.yuv'%(i, orig_img.shape[0], orig_img.shape[1])
            
            np_proto.tofile(path_proto)            
            np_pred.tofile(path_pred)   
            orig_img.tofile(path_orig_img)   
            
            print('\n[LOOP_%d INFO FINISH]----------------------------------------------------------------\n'%i)
          
            print('img.shape[2:]:', img.shape[2:])  # from dim2 to last dimension
            print('img.shape[1:]:', img.shape[1:])  # from dim1 to last dimension
            print('img.shape[0:]:', img.shape[0:])  # from dim0 to last dimension
        
            if not len(pred):  # save empty boxes
                results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6]))
                continue
            
            if self.args.retina_masks:
                if not isinstance(orig_imgs, torch.Tensor):
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                    
                # print('retina_masks_true pred:', len(pred))
                masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC
            else:
                # print('retina_masks_false pred:', len(pred))
                
                masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
                
                print('\nmasks:', masks.shape)
                
                if not isinstance(orig_imgs, torch.Tensor):
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks))
        
        return results


def predict(cfg=DEFAULT_CFG, use_python=False):
    """Runs YOLO object detection on an image or video source."""
    model = cfg.model or 'yolov8n-seg.pt'
    source = cfg.source if cfg.source is not None else ROOT / 'assets' if (ROOT / 'assets').exists() \
        else 'https://ultralytics.com/images/bus.jpg'

    args = dict(model=model, source=source)
    
    print('use_python: %s' % use_python, 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    
    if use_python:
        from ultralytics import YOLO
        YOLO(model)(**args)
    else:
        predictor = SegmentationPredictor(overrides=args)
        predictor.predict_cli()


if __name__ == '__main__':
    logger.debug("This is a debug log")
    logger.info("This is an info log")
    logger.warning("This is a warning log")
    logger.error("This is an error log")
    logger.critical("This is a critical log")

    predict()

# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch

from ultralytics.yolo.engine.predictor_multi import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops


class MultiPredictor(BasePredictor):

    def postprocess_det(self, preds, img, orig_imgs):
        """Postprocesses predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            path = self.batch[0]
            img_path = path[i] if isinstance(path, list) else path
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

    def postprocess_seg(self, preds, img=None, orig_imgs=None):
        """Turn segmentation logits into a class-index mask at the original image size.

        lanedet change: the original resized straight to a hard-coded 1280x720 (BDD100K's size), which
        misaligns masks for any other camera and ignores the letterbox padding. Here we crop the padding
        away first, then resize to the real image size.
        """
        if img is not None and orig_imgs is not None:
            h_in, w_in = img.shape[2:]
            orig = orig_imgs[0] if isinstance(orig_imgs, list) else orig_imgs
            h0, w0 = orig.shape[:2]
            gain = min(h_in / h0, w_in / w0)
            pad_w, pad_h = (w_in - w0 * gain) / 2, (h_in - h0 * gain) / 2
            # preds may be at a different resolution than the input (seg head outputs at input size here)
            sy, sx = preds.shape[2] / h_in, preds.shape[3] / w_in
            top, bottom = int(round((pad_h - 0.1) * sy)), int(round((h_in - pad_h + 0.1) * sy))
            left, right = int(round((pad_w - 0.1) * sx)), int(round((w_in - pad_w + 0.1) * sx))
            preds = preds[:, :, max(top, 0):bottom, max(left, 0):right]
            size = (h0, w0)
        else:
            size = (720, 1280)  # original behaviour
        preds = torch.nn.functional.interpolate(preds, size=size, mode='bilinear', align_corners=False)
        preds = self.sigmoid(preds)
        _, preds = torch.max(preds, 1)
        return preds


def predict(cfg=DEFAULT_CFG, use_python=False):
    """Runs YOLO model inference on input image(s)."""
    model = cfg.model or 'yolov8n.pt'
    source = cfg.source if cfg.source is not None else ROOT / 'assets' if (ROOT / 'assets').exists() \
        else 'https://ultralytics.com/images/bus.jpg'

    args = dict(model=model, source=source)
    if use_python:
        from ultralytics import YOLO
        YOLO(model)(**args)
    else:
        predictor = MultiPredictor(overrides=args)
        predictor.predict_cli()


if __name__ == '__main__':
    predict()

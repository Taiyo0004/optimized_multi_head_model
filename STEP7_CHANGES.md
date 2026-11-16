# Step 7 (code): three-class lane head - background, solid, dashed

## Label format (what the Step 5 converter will write)
Lane label files stay YOLO polygons, one line per lane marking:
`17 x1 y1 x2 y2 ...` = solid, `18 x1 y1 ...` = dashed (normalised). A dashed lane is one continuous
polygon along the whole line (gaps included), the same way BDD100K labels it. Polygons keep mosaic,
scale and perspective augmentation working on lanes.

## Changes
| File | Change |
|---|---|
| `models/v8/yolov8-lanedet-s.yaml` | Lane `Segment` nc 1 -> 2 (3 output channels); `tnc` 19 |
| `datasets/lanedet.yaml` | `nc_list: [17, 2]`, `map: [None, {'17': '0', '18': '1'}]`, names `lane_solid`, `lane_dashed` |
| `yolo/data/base.py` | `update_multi_labels` applies the yaml `map` per task (lane 17/18 -> 0/1); `single_cls` never touches lane classes |
| `yolo/data/augment.py` | `Format` builds a class-valued mask for seg tasks (0 = background, 1 = solid, 2 = dashed) instead of an instance-index mask that the loss then collapsed to "line" |
| `yolo/data/dataset.py` | validation transforms use the class-valued mask too |
| `yolo/v8/DecSeg/train.py` | New `SegLoss.multiclass`: weighted cross-entropy (line classes x `seg_line_w`) + Dice over line classes. Used automatically when the head has > 2 channels; the binary A-YOLOM loss is untouched. `lane_lambda` scales the lane loss (Section 7.3). Loss names become `dice_loss`, `ce_loss` |
| `yolo/v8/DecSeg/val.py` | Multi-class lane metrics (below); validation plots colour each lane class; log rows labelled by task |
| `yolo/engine/validator.py` | Lane metrics and val losses are returned so they reach the CSV |
| `yolo/engine/trainer.py` | One results CSV per task (`results_detection-object.csv`, `results_seg-lane.csv`) - before, both tasks wrote rows with different columns into one file |
| `yolo/cfg/default.yaml` | New: `seg_ce: 1.0`, `seg_dice: 1.0`, `seg_line_w: 5.0`, `lane_lambda: 1.0`. Changed: `fliplr: 0.0` (no flips - speed-sign digits) |

## Lane metrics
| Metric | Meaning | Comparable to A-YOLOM? |
|---|---|---|
| `IoU` | any line vs background, per image, averaged | yes - A-YOLOM's lane IoU |
| `subacc` | any line vs background, balanced accuracy | yes - A-YOLOM's lane accuracy |
| `pixacc` | 3-class pixel accuracy | new |
| `mIoU` | mean IoU over classes present in each image | new |
| `IoU_lane_solid`, `IoU_lane_dashed` | per class, averaged over images where the class appears | new |
Fitness (best.pt selection) = detection fitness + `IoU` + `mIoU`, as before.

## Weight start
The lane head's last layer (2 -> 3 channels) is warm-started from A-YOLOM: background <- background,
solid <- line, dashed <- line. So the model starts out finding lane lines like A-YOLOM and only has to
learn solid vs dashed. `weights/lanedet_s_init.pt` is rebuilt by `step7_check.py`.

## Known cosmetic issue
Training-batch preview images (`train_batch*.jpg`) colour lane masks by class value rather than per
instance - harmless.

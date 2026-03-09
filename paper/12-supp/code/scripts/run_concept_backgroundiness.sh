
CLASSES=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)
LAYER=(encoder.features.0 encoder.features.5 encoder.features.10
 )
for s in "${CLASSES[@]}"
do
  for l in "${LAYER[@]}"
  do
    python3 -m experiments.concept_backgroundiness --model_name unet --dataset_name cityscapes --layer_name $l --class_id $s
    python3 -m experiments.concept_backgroundiness_evaluation --model_name unet --dataset_name cityscapes --layer_name $l --class_id $s
  done
done

CLASSES=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)
LAYER=(backbone.layer4.2.conv2 backbone.layer4.0.conv3 backbone.layer3.0.conv1)
for s in "${CLASSES[@]}"
do
  for l in "${LAYER[@]}"
  do
    python3 -m experiments.concept_backgroundiness --model_name deeplabv3plus --dataset_name voc2012 --layer_name $l --class_id $s  --batch_size 7
    python3 -m experiments.concept_backgroundiness_evaluation --model_name deeplabv3plus --dataset_name voc2012 --layer_name $l --class_id $s
  done
done


CLASSES=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79)
LAYER=(backbone.ERBlock_5.0.rbr_dense.conv backbone.ERBlock_3.0.rbr_dense.conv backbone.ERBlock_4.0.rbr_dense.conv)
for s in "${CLASSES[@]}"
do
  for l in "${LAYER[@]}"
  do
    python3 -m experiments.concept_backgroundiness --model_name yolov6 --dataset_name coco2017 --layer_name $l --class_id $s --batch_size 7
    python3 -m experiments.concept_backgroundiness_evaluation --model_name yolov6 --dataset_name coco2017 --layer_name $l --class_id $s
  done
done
#
CLASSES=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79)
LAYER=(model.6.cv3.conv model.8.cv3.conv model.10.conv)
for s in "${CLASSES[@]}"
do
  for l in "${LAYER[@]}"
  do
    python3 -m experiments.concept_backgroundiness --model_name yolov5 --dataset_name coco2017 --layer_name $l --class_id $s
    python3 -m experiments.concept_backgroundiness_evaluation --model_name yolov5 --dataset_name coco2017 --layer_name $l --class_id $s
  done
done

for rel_init in {ones,prob,logits};do
  python3 -m experiments.explanation_complexity --model_name unet --dataset_name cityscapes --rel_init $rel_init
done
python3 -m experiments.explanation_complexity --model_name deeplabv3plus --dataset_name voc2012
python3 -m experiments.explanation_complexity --model_name yolov6 --dataset_name coco2017
python3 -m experiments.explanation_complexity --model_name yolov5 --dataset_name coco2017

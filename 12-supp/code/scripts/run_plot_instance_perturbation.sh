
for rel_init in {ones,prob,logits};do
  python3 -m experiments.plot_instance_perturbation --model_name deeplabv3plus --dataset_name voc2012 --insertion True --rel_init $rel_init
  python3 -m experiments.plot_instance_perturbation --model_name deeplabv3plus --dataset_name voc2012 --insertion False --rel_init $rel_init
  python3 -m experiments.plot_instance_perturbation --model_name unet --dataset_name cityscapes --insertion False --rel_init $rel_init
  python3 -m experiments.plot_instance_perturbation --model_name unet --dataset_name cityscapes --insertion True --rel_init $rel_init
done
#
#
python3 -m experiments.plot_instance_perturbation --model_name yolov6 --dataset_name coco2017 --insertion False
python3 -m experiments.plot_instance_perturbation --model_name yolov5 --dataset_name coco2017 --insertion False
#
python3 -m experiments.plot_instance_perturbation --model_name yolov6 --dataset_name coco2017 --insertion True
python3 -m experiments.plot_instance_perturbation --model_name yolov5 --dataset_name coco2017 --insertion True
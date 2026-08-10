
LAYERS=(backbone.stem.rbr_reparam backbone.ERBlock_2.1.conv1.rbr_1x1.conv backbone.ERBlock_3.0.rbr_1x1.conv backbone.ERBlock_3.1.block.0.rbr_1x1.conv backbone.ERBlock_3.1.block.2.rbr_1x1.conv backbone.ERBlock_4.1.conv1.rbr_1x1.conv backbone.ERBlock_4.1.block.1.rbr_1x1.conv backbone.ERBlock_4.1.block.3.rbr_1x1.conv backbone.ERBlock_5.0.rbr_1x1.conv backbone.ERBlock_5.1.block.0.rbr_1x1.conv neck.Rep_p4.conv1.rbr_1x1.conv neck.Rep_p4.block.1.rbr_1x1.conv neck.Rep_p3.conv1.rbr_1x1.conv neck.Rep_p3.block.1.rbr_1x1.conv neck.Rep_n3.conv1.rbr_1x1.conv neck.Rep_n3.block.1.rbr_1x1.conv neck.Rep_n4.conv1.rbr_1x1.conv neck.Rep_n4.block.1.rbr_1x1.conv neck.reduce_layer1.conv detect.cls_convs.1.conv detect.reg_convs.2.conv detect.reg_preds.0 detect.obj_preds.1 detect.stems.2.conv)
for l in "${LAYERS[@]}"
do
  python3 -m experiments.instance_perturbation_od --model_name yolov6 --dataset_name coco2017 --layer_name $l --num_samples 100 --batch_size 20 --insertion True
  python3 -m experiments.instance_perturbation_od --model_name yolov6 --dataset_name coco2017 --layer_name $l --num_samples 100 --batch_size 20 --insertion False
done

LAYERS=(model.0.conv model.2.cv3.conv model.2.m.1.cv2.conv model.4.cv3.conv model.4.m.1.cv2.conv model.4.m.3.cv2.conv model.6.cv3.conv model.6.m.1.cv2.conv model.6.m.3.cv2.conv model.6.m.5.cv2.conv model.8.cv3.conv model.8.m.1.cv2.conv model.13.cv1.conv model.13.m.0.cv2.conv model.17.cv1.conv model.17.m.0.cv2.conv model.20.cv1.conv model.20.m.0.cv2.conv model.23.cv1.conv model.23.m.0.cv2.conv model.24.m.1
)
for l in "${LAYERS[@]}"
do
  python3 -m experiments.instance_perturbation_od --model_name yolov5 --dataset_name coco2017 --layer_name $l --num_samples 100 --batch_size 20 --insertion False
  python3 -m experiments.instance_perturbation_od --model_name yolov5 --dataset_name coco2017 --layer_name $l --num_samples 100 --batch_size 20 --insertion True
done


#
for rel_init in {prob,ones,logits};do
  LAYERS=(
  encoder.features.7 encoder.features.0 encoder.features.2 encoder.features.5  encoder.features.10 encoder.features.12 encoder.features.15 encoder.features.17 encoder.features.20 encoder.features.22 decoder.center.0.0 decoder.center.1.0
   decoder.blocks.0.conv1.0 decoder.blocks.0.conv2.0 decoder.blocks.1.conv1.0 decoder.blocks.1.conv2.0 decoder.blocks.2.conv1.0 decoder.blocks.2.conv2.0 decoder.blocks.3.conv1.0 decoder.blocks.3.conv2.0 decoder.blocks.4.conv1.0 decoder.blocks.4.conv2.0 segmentation_head.0)
  for l in "${LAYERS[@]}"
  do
    echo "run ${rel_init} ${l}"
    python3 -m experiments.instance_perturbation --model_name unet --dataset_name cityscapes --layer_name $l --num_samples 100 --batch_size 20 --insertion True --rel_init $rel_init
    python3 -m experiments.instance_perturbation --model_name unet --dataset_name cityscapes --layer_name $l --num_samples 100 --batch_size 20 --insertion False --rel_init $rel_init
  done
done

for rel_init in {prob,ones,logits};do
  echo "run ${rel_init}"
  LAYERS=(
  backbone.conv1 backbone.layer1.0.conv3 backbone.layer1.1.conv2 backbone.layer1.2.conv2 backbone.layer2.0.conv2 backbone.layer2.1.conv1 backbone.layer2.2.conv1 backbone.layer2.3.conv1
  backbone.layer3.0.conv1 backbone.layer3.0.downsample.0  backbone.layer3.1.conv3
  backbone.layer3.2.conv3 backbone.layer3.3.conv3 backbone.layer3.4.conv3
  backbone.layer3.5.conv3 backbone.layer4.0.conv3 backbone.layer4.1.conv2 backbone.layer4.2.conv2 classifier.aspp.convs.0.0
     classifier.aspp.convs.3.0 classifier.classifier.0)
  for l in "${LAYERS[@]}"
  do
    echo "run ${rel_init} ${l}"
    python3 -m experiments.instance_perturbation --model_name deeplabv3plus --dataset_name voc2012 --layer_name $l --num_samples 100 --batch_size 20 --insertion False --rel_init $rel_init
    python3 -m experiments.instance_perturbation --model_name deeplabv3plus --dataset_name voc2012 --layer_name $l --num_samples 100 --batch_size 20 --insertion True --rel_init $rel_init
  done
done
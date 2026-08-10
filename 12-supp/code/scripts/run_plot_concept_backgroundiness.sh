python3 -m experiments.plot_concept_backgroundiness --model_name unet --dataset_name cityscapes --labels "CRP Relevance,LRP,Activation"
python3 -m experiments.plot_concept_backgroundiness --model_name deeplabv3plus --dataset_name voc2012 --labels "CRP Relevance,LRP,Activation"
python3 -m experiments.plot_concept_backgroundiness --model_name yolov6 --dataset_name coco2017 --labels "CRP Relevance,LRP,Activation"
python3 -m experiments.plot_concept_backgroundiness --model_name yolov5 --dataset_name coco2017 --labels "CRP Relevance,LRP,Activation"

python3 -m experiments.plot_concept_backgroundiness --model_name unet --dataset_name cityscapes --labels "CRP Relevance,Guided GradCAM,GradCAM,SSGradCAM"
python3 -m experiments.plot_concept_backgroundiness --model_name deeplabv3plus --dataset_name voc2012 --labels "CRP Relevance,Guided GradCAM,GradCAM,SSGradCAM"
python3 -m experiments.plot_concept_backgroundiness --model_name yolov6 --dataset_name coco2017 --labels "CRP Relevance,Guided GradCAM,GradCAM,SSGradCAM"
python3 -m experiments.plot_concept_backgroundiness --model_name yolov5 --dataset_name coco2017 --labels "CRP Relevance,Guided GradCAM,GradCAM,SSGradCAM"


CLASSES=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80) #
for s in "${CLASSES[@]}"
do
  #python3 -m experiments.global_class_concepts --model_name yolov6 --dataset_name coco2017 --class_id $s --batch_size 4
  #python3 -m experiments.global_class_concepts --model_name yolov5 --dataset_name coco2017 --class_id $s --batch_size 4
done

for rel_init in {prob,ones};do
  #echo "run ${rel_init}"
  #CLASSES=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)
  #for s in "${CLASSES[@]}"
  #do
    #python3 -m experiments.global_class_concepts --model_name deeplabv3plus --dataset_name voc2012 --class_id $s --batch_size 3 --rel_init $rel_init
  #done

  CLASSES=(0 1) #
  for s in "${CLASSES[@]}"
  do
    python3 -m experiments.global_class_concepts --model_name pidnet --dataset_name flood --class_id $s --batch_size 5 --rel_init $rel_init
  done
done

CLASSES=(0 1 ) #

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
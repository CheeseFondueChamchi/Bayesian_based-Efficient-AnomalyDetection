while true
do
  python train.py --config 'kt_config.yaml' --loglevel 'info' --ru_ids_path Target_RU_list_1_new.csv
  #python train_xgboost.py --config 'kt_config.yaml'
  sleep 15
done


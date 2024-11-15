# ai-server-id 1: Target_RU_list_1_wacs2.csv, tbl_ai_output_1, ip:
# ai-server-id 2: Target_RU_list_2_wacs3.csv, tbl_ai_output_2, ip:

while true
do
  python inference.py --config 'kt_config.yaml' --ai-server-id 1 --loglevel 'info' --ru_ids_path Target_RU_list_1_new.csv
  sleep 15
done

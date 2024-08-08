# Define a dictionary with datasets as keys and data splits as values
declare -A dataset_splits
dataset_splits["dataset_07-16_10-31-09"]="d3_physical_container d5_physical_tool d5_sports_equipment"
#dataset_splits["dataset_07-16_10-31-09"]="d1_root d2_n1 d3_n2 d3_physical_container d4_concept d4_equipment d4_food d4_n3 d4_product d4_utensil d4_vehicle d5_animal d5_appliance d5_equipment d5_food d5_machine d5_n4 d5_physical_tool d5_product d5_sports_equipment d5_utensil d5_vehicle"
#dataset_splits["dataset_08-02_13-47-21"]="d1_root d2_animal d2_appliance d2_bag d2_cooking_appliance d2_dish d2_equipment d2_food d2_fruit d2_furniture d2_machine d2_n1 d2_plant d2_plumbing_fixture d2_product d2_seat d2_skiing_and_snowboarding_equipment d2_sports_equipment d2_tableware d2_tool d2_vehicle d2_vessel d2_wheeled_vehicle"

for dataset in "${!dataset_splits[@]}"
do
    splits=${dataset_splits[$dataset]}
    for split in $splits
    do
       python baseline.py --datasplit $split --dataset $dataset
    done
done

dataset=busi
input_size=256
python train_mk.py --arch MK_UNet --dataset ${dataset} --input_w ${input_size} --input_h ${input_size} --name ${dataset}_UKAN  --data_dir ./inputs
python val.py --name ${dataset}_UKAN 

dataset=glas
input_size=512
python train_mk.py --arch MK_UNet --dataset ${dataset}  --batch_size 4 --input_w ${input_size} --input_h ${input_size} --name ${dataset}_UKAN  --data_dir ./inputs
python val.py --name ${dataset}_UKAN 

dataset=cvc
input_size=256
python train_mk.py --arch MK_UNet --dataset ${dataset} --input_w ${input_size} --input_h ${input_size} --name ${dataset}_UKAN  --data_dir ./inputs
python val.py --name ${dataset}_UKAN 

dataset=oct
input_size=256
python train_mk.py --arch MK_UNet --dataset ${dataset} --input_w ${input_size} --input_h ${input_size} --name ${dataset}_UKAN  --data_dir ./inputs
python val.py --name ${dataset}_UKAN 


dataset=isic18
input_size=256
python train_mk.py --arch MK_UNet --dataset ${dataset} --input_w ${input_size} --input_h ${input_size} --name ${dataset}_UKAN  --data_dir ./inputs
python val.py --name ${dataset}_UKAN 

dataset=fives
input_size=256
python train_mk.py --arch MK_UNet --dataset ${dataset} --input_w ${input_size} --input_h ${input_size} --name ${dataset}_UKAN  --data_dir ./inputs
python val.py --name ${dataset}_UKAN 
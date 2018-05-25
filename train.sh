CUDA_VISIBLE_DEVICES=7 nohup python train.py \
--trainlist=tool/train_zhengzhi.txt \
--vallist=tool/train_zhengzhi.txt \
--lang --cuda --random_sample \
--displayInterval=100 \
--valInterval=4000 \
--saveInterval=4000 \
--lr=0.001 \
--keep_ratio \
--experiment=24_1bi_sru \
>24_1bi_sru_MLT.log 2>&1 &

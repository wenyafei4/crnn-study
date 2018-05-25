UDA_VISIBLE_DEVICES=7 python val_CTC.py --lang --experiment expr_basic_lang --trainlist tool/train_zhengzhi.txt --vallist tool/testComplex.txt --cuda --adam --lr=0.001 --saveInterval=4000 --displayInterval=100 --valInterval=1000 --random_sample --keep_ratio \
--crnn=24_1bi_sru/netCRNN_199_4000.pth 

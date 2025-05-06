python3 -m torch.distributed.launch --nproc_per_node=1 \
/home/a615/tianzhaonan/model-pc_cross/train.py \
--do_train --num_thread_reader=0 \
--epochs=100 --batch_size=64 \
--n_display=10 \
--data_path /home/a615/tianzhaonan/model-pc_cross/train_valid_test.pt \
--output_dir /home/a615/tianzhaonan/model-pc_cross/output_align \
--lr 5e-5  \
--visual_num_hidden_layers 4 \
--bert_num_hidden_layers 6 \
--audio_num_hidden_layers 4 --aligned

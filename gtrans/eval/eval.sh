data_name="Hoppity"
#cooked_root="/storage/hdd/chontipan/cooked-full-fmt-shift_node/"
cooked_root="/storage/hdd/chontipan/cooked-no-op-fmt-shift_node"
save_dir="/storage/hdd/chontipan/hoppity/save_dir"
target_model="/storage/hdd/chontipan/no-diff.ckpt"
eval_dump_folder="/storage/hdd/chontipan/hoppity/eval_dump/"
#export CUDA_VISIBLE_DEVICES=0

python eval.py \
	-target_model $target_model \
	-data_root $cooked_root \
	-data_name $data_name \
	-save_dir $save_dir \
	-iters_per_val 100 \
	-beam_size 3 \
	-batch_size 10 \
	-topk 3 \
	-gnn_type 's2v_multi' \
	-max_lv 4 \
	-max_modify_steps 1 \
	-gpu 2 \
	-resampling True \
	-comp_method "bilinear" \
	-bug_type True \
	-loc_acc True \
	-val_acc True \
	-output_all True \
	-loc_given True \
	$@

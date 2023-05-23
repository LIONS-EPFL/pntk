#!/bin/bash
modelname=context_norm_poly
outdir=res/scale/poly
for r in {1..8}
do
	python3 ./trainer.py --run $r --model_name $modelname --outdir $outdir --dset_names scale_train scale_test1 scale_test2 scale_test3 scale_test4 scale_test5 #--train_steps 2 --save_checkpoint_steps 1
	for t in {0..5}
	do
		python3 ./eval.py --run $r --test_set_ind $t --model_name $modelname --outdir $outdir --test_set_names scale_train scale_test1 scale_test2 scale_test3 scale_test4 scale_test5 #--checkpoint 1
	done
done

temp="${outdir}/eval"
python3 ./printresult.py --dirs $temp

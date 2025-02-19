
## Command to train
```
PYTHONPATH='.../Diabetic-Retinopathy' \
CUDA_VISIBLE_DEVICES=0 \
taskset --cpu-list 0-9 \
python train.py --config multiclass_model.yml
```
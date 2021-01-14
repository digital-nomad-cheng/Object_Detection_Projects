
python demo/demo.py --config-file configs/retinaface_R50_FPN.yaml --input datasets/widerface/val/images/*/*.jpg --output work_dirs/retinaface_R50_FPN/val  --opts MODEL.WEIGHTS work_dirs/retinaface_R50_FPN/model_final.pth
python evaluate_widerface/parse_predictions_to_widerface_val.py --res work_dirs/retinaface_R50_FPN/val/results.pkl --save work_dirs/retinaface_R50_FPN/widerface_val

cd evaluate_widerface/widerface_evaluate/
python evaluation.py --p ../../work_dirs/retinaface_R50_FPN/widerface_val/

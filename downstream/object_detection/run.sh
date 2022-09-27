export DETECTRON2_DATASETS=/media/data1/binh/detectron2/

# good results for BYOL
# python3 train_object_detection.py --config-file configs/pascal_voc_R_50_C4_24k_moco.yaml\
#     --num-gpus 8 MODEL.WEIGHTS /media/data1/binh/solo-learn/byol/3ividqq7/detectron_model.pkl SOLVER.IMS_PER_BATCH 16 SOLVER.BASE_LR 0.1\
#     OUTPUT_DIR /media/data1/binh/solo-learn/byol/3ividqq7/output/

# # mocov2plus
# python3 train_object_detection.py --config-file configs/pascal_voc_R_50_C4_24k_moco.yaml\
#     --num-gpus 8 \
#     MODEL.WEIGHTS /media/data1/binh/solo-learn/imagenet/barlowtwin/abcd1234/detectron_model.pkl \
#     SOLVER.IMS_PER_BATCH 16 \
#     SOLVER.BASE_LR 0.1\
#     OUTPUT_DIR /media/data1/binh/solo-learn/imagenet/barlowtwin/abcd1234/output/

# python3 train_object_detection.py --config-file configs/pascal_voc_R_50_C4_24k_moco.yaml\
#     --num-gpus 8 \
#     MODEL.WEIGHTS /media/data1/binh/solo-learn/imagenet/mocov2plus/abcd1234/detectron_model.pkl \
#     SOLVER.IMS_PER_BATCH 16 \
#     SOLVER.BASE_LR 0.1\
#     OUTPUT_DIR /media/data1/binh/solo-learn/imagenet/mocov2plus/abcd1234/output/

# python3 train_object_detection.py --config-file configs/pascal_voc_R_50_C4_24k_moco.yaml\
#     --num-gpus 8 \
#     MODEL.WEIGHTS /media/data1/binh/simsiam/Simsiam/results/simsiam_detectron_model.pkl \
#     SOLVER.IMS_PER_BATCH 16 \
#     SOLVER.BASE_LR 0.1 \
#     OUTPUT_DIR /media/data1/binh/simsiam/Simsiam/results/output/


# python3 train_object_detection.py --config-file configs/pascal_voc_R_50_C4_24k_moco.yaml\
#     --num-gpus 8 \
#     MODEL.WEIGHTS /media/data1/binh/solo-learn/imagenet/byol/abcd1234/detectron_model.pkl \
#     SOLVER.IMS_PER_BATCH 16 SOLVER.BASE_LR 0.1 \
#     OUTPUT_DIR /media/data1/binh/solo-learn/imagenet/byol/abcd1234/output/





# python3 train_object_detection.py --config-file configs/pascal_voc_R_50_C4_24k_moco.yaml\
#     --num-gpus 8 MODEL.WEIGHTS /media/data1/binh/solo-learn/imagenet/byol/offline-h41nev57/detectron_model.pkl \
#     SOLVER.IMS_PER_BATCH 16 \
#     SOLVER.BASE_LR 0.15 \
#     OUTPUT_DIR /media/data1/binh/solo-learn/imagenet/byol/offline-h41nev57/output/


python3 train_object_detection.py --config-file configs/pascal_voc_R_50_C4_24k_moco.yaml \
    --num-gpus 6 MODEL.WEIGHTS /media/data1/binh/solo-learn/imagenet/vicreg/detectron_model.pkl \
    SOLVER.IMS_PER_BATCH 18 \
    SOLVER.BASE_LR 0.1 \
    OUTPUT_DIR /media/data1/binh/solo-learn/imagenet/vicreg/output/
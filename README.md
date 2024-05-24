# GRM_layer
Weakly supervised semantic segmentation by knowledge graph

Note:The external knowledge graph CM_kg_57_info.json obtained by ConceptNet. It contains the relationship matrix of PASCAL VOC 20 classes (20×20) and MSCOCO 80 classes (80×80).


Python 3.6, PyTorch 1.9, and others in environment.yml
You can create the environment from environment.yml file
conda env create -f environment.yml

Usage (PASCAL VOC)

Step 1. Prepare dataset.
Download PASCAL VOC 2012 devkit from official website.
You need to specify the path ('voc12_root') of your downloaded devkit in the following steps.

Step 2. Train ReCAM and generate seeds.
python run_sample.py --voc12_root ./VOCdevkit/VOC2012/ --work_space YOUR_WORK_SPACE --train_cam_pass True --train_recam_pass True --make_recam_pass True --eval_cam_pass True 

Step 3. Train IRN and generate pseudo masks.
python run_sample.py --voc12_root ./VOCdevkit/VOC2012/ --work_space YOUR_WORK_SPACE --cam_to_ir_label_pass True --train_irn_pass True --make_sem_seg_pass True --eval_sem_seg_pass True 

Step 4. Train semantic segmentation network.
To train DeepLab-v2

Usage (MS COCO)
Step 1. Prepare dataset.
Download MS COCO images from the official COCO website.
Generate mask from annotations (annToMask.py file in ./mscoco/).
Download MS COCO image-level labels from here and put them in ./mscoco/

Step 2. Train ReCAM and generate seeds.
python run_sample_coco.py --mscoco_root ../MSCOCO/ --work_space YOUR_WORK_SPACE --train_cam_pass True --train_recam_pass True --make_recam_pass True --eval_cam_pass True

Step 3. Train IRN and generate pseudo masks.
python run_sample_coco.py --mscoco_root ../MSCOCO/ --work_space YOUR_WORK_SPACE --cam_to_ir_label_pass True --train_irn_pass True --make_sem_seg_pass True --eval_sem_seg_pass True 

Step 4. Train semantic segmentation network.

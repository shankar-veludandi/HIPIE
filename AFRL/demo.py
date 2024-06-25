# Import necessary libraries
import os
import torch
from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
from detectron2.utils.visualizer import Visualizer
from detectron2.projects.hipie.demo_lib.demo_utils import *
from scipy.ndimage import distance_transform_edt, label as ndi_label
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from detectron2.data.detection_utils import read_image, convert_PIL_to_numpy
from IPython.display import display

# Set up paths and arguments
args = Namespace()
args.config_file = 'projects/HIPIE/configs/eval/image_joint_vit_huge_32g_pan_maskdino_ade_test.yaml'
args.opts = ['OUTPUT_DIR','outputs/test_maskdino_pan_fixed_lan']
args.task = "detection"

# Set up the configuration
cfg = setup_cfg(args,'weights/R-50.pkl')
cfg.MODEL.CLIP.ALPHA = 0.2
cfg.MODEL.CLIP.BETA = 0.45
cfg.MODEL.PANO_TEMPERATURE_CLIP_FG = 0.01
cfg.MODEL.PANO_TEMPERATURE = 0.06

# Initialize demo and model
demo, model = init_demo(cfg)

# Set up the parts demo
args = Namespace()
args.config_file = 'projects/HIPIE/configs/eval/image_joint_r50_pan_maskdino_parts.yaml'
args.opts = ['OUTPUT_DIR','outputs/parts']
args.task = "detection"
cfg = setup_cfg(args, 'weights/R-50.pkl')
demo_parts = VisualizationDemo(cfg)

# Set up SAM
model_type = "vit_h"
sam_checkpoint = 'weights/sam_vit_h_4b8939.pth'
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
device = "cuda"
sam.to(device)
mask_generator = SamAutomaticMaskGenerator(sam)
predictor = SamPredictor(sam)

# Process the image
img_path = 'assets/demo_sam1.jpg'
out_path = 'outputs/demo'

#  Open the image file
image = Image.open(img_path)

#Downscale the image to reduce memory usage
min_l = min(image.width, image.height)
#image = image.crop((0,0,min_l,min_l)).resize((1500,1500))
image = image.crop((0,0,min_l,min_l)).resize((800,800)) #Resize to smaller dimensions to save memory

# Convert the image to numpy arrays in different formats
img = convert_PIL_to_numpy(image, format="BGR")
img_rgb = convert_PIL_to_numpy(image, format="RGB")

# Extract the filename without the extension
fname = img_path.split('/')[-1].split('.')[0]
os.makedirs(out_path, exist_ok=True)

# Metadata configuration for panoptic segmentation
meta_data_key = dict(
    coco_panoptic='coco_2017_train_panoptic_with_sem_seg',
    #ade20k_150='ade20k_panoptic_val',
    #ade20k_847='ade20k_full_sem_seg_val',
    #pascal_context_59='ctx59_sem_seg_val',
    #pascal_context_459='ctx459_sem_seg_val',
    #pascal_voc_21='pascal21_sem_seg_val',
)
name_short = 'coco_panoptic'
name = meta_data_key[name_short]
metadata = MetadataCatalog.get(name)

# Generate category to index mapping
cat2ind = cat2ind_panoptics_coco(get_openseg_labels(name_short), name)
thing_class_ids = metadata.thing_dataset_id_to_contiguous_id.values()
is_thing = {k: (k-1 in thing_class_ids) for k,v in cat2ind.items()}
is_thing[0] = False # BG

# Get segment labels
open_seg_labels = get_openseg_labels(name_short, prompt_engineered=True)
open_seg_labels_no_prompt = get_openseg_labels(name_short, prompt_engineered=False)
test_args = dict(
    test_categories=open_seg_labels_no_prompt,
    open_seg_labels=open_seg_labels,
    test_is_thing=is_thing,
)

# Run the demo on the image with panoptic segmentation
test_args_custom = test_args
predictions, visualized_output = demo.run_on_image(img, 0.5, args.task, dataset_name='coco_panoptic', **test_args_custom) # removed visualized_output variable

# Move intermediate results to CPU to free up GPU memory
with torch.no_grad():
    panoptic_seg, segments_info = predictions['panoptic_seg']
panoptic_seg = panoptic_seg.cpu()

# Run parts demo on the image
predictions, visualized_output = demo_parts.run_on_image(img, 0.5, args.task, None, **get_args_eval())  # removed visualized_output variable

# Extract segmentation results to convert to CPU
parts_seg = predictions['sem_seg'].cpu().argmax(0)

# Generate part instance masks
parts_seg_instance, parts_seg_instance_cls = sem_to_instance_map(panoptic_seg, segments_info, parts_seg, test_args, max_id=200)

# Visualize and save panoptic segmentation result
vis = Visualizer(img, metadata=metadata)
vis.draw_panoptic_seg(panoptic_seg.cpu(), segments_info)
display(vis.get_output().fig)
vis.get_output().save(os.path.join(out_path, f'{fname}_pano.jpg'))

# Merge part and panoptic segmentation masks
masks_vv, labels_vv = merge_part_and_pano(parts_seg_instance, parts_seg_instance_cls, panoptic_seg, test_args, segments_info)
cliped_parts = torch.clip(torch.sum(torch.stack(parts_seg_instance), dim=0), 0, 1).cpu()
labels_sam = [{'id':i+1,'name':k}for i,k in enumerate(labels_vv)]
input_sam = torch.stack(masks_vv).cpu()
new_input_sam = input_sam
new_labels_sam = labels_sam

#Generate SAM masks
masks_sam = mask_generator.generate(img)
mask_sam_stack = np.stack([x['segmentation'].astype(float) for x in masks_sam])
mask_sam_stack = torch.tensor(mask_sam_stack).float().to(new_input_sam)

# Prepare for voting
sem_seg_with_bg = torch.cat([new_input_sam[0].unsqueeze(0)*0.0, new_input_sam / (new_input_sam.sum(0, keepdims=True) + 1e-5)], dim=0)
bg_confidence = 0.05
sem_seg_with_bg[0] = bg_confidence
voting_output = vote(mask_sam_stack, sem_seg_with_bg.to('cpu'), img.shape[:2])

# Map labels
idx_2_name = cat2ind_panoptics_coco(new_labels_sam)
mask_labels = voting_output['mask_labels'].cpu().numpy()
final_sam_masks = mask_sam_stack.cpu()

# Filter masks
final_sam_masks = final_sam_masks[mask_labels > 0]
mask_labels = mask_labels[mask_labels > 0]
mask_labels_text = [idx_2_name[k] for k in mask_labels]
mask_labels_text = [x.split(',')[0] for x in mask_labels_text]

# Plot SAM masks
plt.figure(figsize=(14,14))
plt.imshow(img_rgb, alpha=0.7)
show_anns(masks_sam)
plt.axis('off')
plt.savefig(os.path.join(out_path, fname + '_sam.jpg'), bbox_inches='tight', pad_inches=0, dpi=600)
plt.show()
plt.close()

# Combine final SAM masks with part segmentation instances
new_final_sam_masks = []
new_mask_labels_text = []
for m in range(len(final_sam_masks)):
    mask = final_sam_masks[m]
    overlap_ratio = (mask * cliped_parts).sum() / (mask.sum() + 1e-9)
    if overlap_ratio < 0.8:
        new_final_sam_masks.append(final_sam_masks[m])
        new_mask_labels_text.append(mask_labels_text[m])
new_final_sam_masks.extend(parts_seg_instance)
new_mask_labels_text.extend(parts_seg_instance_cls)
new_final_sam_masks = torch.stack(new_final_sam_masks).cpu()

# Visualize and save the combined masks
vis = Visualizer(img_rgb)
vis.overlay_instances(masks=new_final_sam_masks, labels=new_mask_labels_text)
vis.get_output().save(os.path.join(out_path, f'{fname}_panpart.jpg'))
display(vis.get_output().fig)

# Save the final image
image.save(os.path.join(out_path, fname + '_image.jpg'))

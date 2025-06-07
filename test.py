import argparse
import os
import json # Added
from os import path as osp # Added for compatibility with copied functions

import torch
from torch import nn
from torch.nn import functional as F
import kornia.geometry as tgm
from kornia.filters.gaussian import GaussianBlur2d # Corrected import path
from torchvision import transforms # Added
from PIL import Image, ImageDraw # Added

# Adjusted imports: VITONDataset and VITONDataLoader are only needed for dataset mode
# from datasets import VITONDataset, VITONDataLoader 
from networks import SegGenerator, GMM, ALIASGenerator
from utils import gen_noise, load_checkpoint, save_images
import numpy as np # Added


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=False, help="Name of the experiment. Required for dataset mode.") # No longer always required

    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('--load_height', type=int, default=1024)
    parser.add_argument('--load_width', type=int, default=768)
    parser.add_argument('--shuffle', action='store_true')

    parser.add_argument('--dataset_dir', type=str, default='./datasets/')
    parser.add_argument('--dataset_mode', type=str, default='test')
    parser.add_argument('--dataset_list', type=str, default='test_pairs.txt')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')
    parser.add_argument('--save_dir', type=str, default='./results/') # Used for dataset mode

    parser.add_argument('--display_freq', type=int, default=1)

    parser.add_argument('--seg_checkpoint', type=str, default='seg_final.pth')
    parser.add_argument('--gmm_checkpoint', type=str, default='gmm_final.pth')
    parser.add_argument('--alias_checkpoint', type=str, default='alias_final.pth')

    # common
    parser.add_argument('--semantic_nc', type=int, default=13, help='# of human-parsing map classes')
    parser.add_argument('--init_type', choices=['normal', 'xavier', 'xavier_uniform', 'kaiming', 'orthogonal', 'none'], default='xavier')
    parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')

    # for GMM
    parser.add_argument('--grid_size', type=int, default=5)

    # for ALIASGenerator
    parser.add_argument('--norm_G', type=str, default='spectralaliasinstance')
    parser.add_argument('--ngf', type=int, default=64, help='# of generator filters in the first conv layer')
    parser.add_argument('--num_upsampling_layers', choices=['normal', 'more', 'most'], default='most',
                        help="If 'more', add upsampling layer between the two middle resnet blocks. "
                             "If 'most', also add one more (upsampling + resnet) layer at the end of the generator.")
    
    # New arguments for API/single image mode
    parser.add_argument('--person_img_path', type=str, default=None, help='Path to the person image for single try-on')
    parser.add_argument('--cloth_img_path', type=str, default=None, help='Path to the cloth image for single try-on')
    parser.add_argument('--output_img_path', type=str, default=None, help='Path to save the output image for single try-on')
    # --cloth_mask_path, --pose_img_path etc. will be derived or assumed based on convention

    opt = parser.parse_args()
    
    # If in API mode, name is not strictly required for directory creation in the same way
    if opt.person_img_path and not opt.name:
        opt.name = "api_tryon" # Default name for API mode if not provided

    if not opt.person_img_path and not opt.name: # name is required for dataset_mode
        parser.error("--name is required when not using --person_img_path (dataset mode)")

    return opt

# Copied and adapted from VITONDataset (datasets.py)
# Ensure these functions get opt for load_width, load_height, semantic_nc if needed, or pass directly
def _get_parse_agnostic(parse, pose_data, load_width, load_height):
    parse_array = np.array(parse)
    parse_upper = ((parse_array == 5).astype(np.float32) +
                   (parse_array == 6).astype(np.float32) +
                   (parse_array == 7).astype(np.float32))
    parse_neck = (parse_array == 10).astype(np.float32)

    r = 10
    agnostic = parse.copy()

    for parse_id, pose_ids in [(14, [2, 5, 6, 7]), (15, [5, 2, 3, 4])]:
        mask_arm = Image.new('L', (load_width, load_height), 'black')
        mask_arm_draw = ImageDraw.Draw(mask_arm)
        i_prev = pose_ids[0]
        for i in pose_ids[1:]:
            if (pose_data[i_prev, 0] == 0.0 and pose_data[i_prev, 1] == 0.0) or \
               (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue
            mask_arm_draw.line([tuple(pose_data[j]) for j in [i_prev, i]], 'white', width=r * 10)
            pointx, pointy = pose_data[i]
            radius = r * 4 if i == pose_ids[-1] else r * 15
            mask_arm_draw.ellipse((pointx - radius, pointy - radius, pointx + radius, pointy + radius), 'white', 'white')
            i_prev = i
        parse_arm = (np.array(mask_arm) / 255) * (parse_array == parse_id).astype(np.float32)
        agnostic.paste(0, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))

    agnostic.paste(0, None, Image.fromarray(np.uint8(parse_upper * 255), 'L'))
    agnostic.paste(0, None, Image.fromarray(np.uint8(parse_neck * 255), 'L'))
    return agnostic

def _get_img_agnostic(img, parse, pose_data, load_width, load_height):
    parse_array = np.array(parse)
    parse_head = ((parse_array == 4).astype(np.float32) +
                  (parse_array == 13).astype(np.float32))
    parse_lower = ((parse_array == 9).astype(np.float32) +
                   (parse_array == 12).astype(np.float32) +
                   (parse_array == 16).astype(np.float32) +
                   (parse_array == 17).astype(np.float32) +
                   (parse_array == 18).astype(np.float32) +
                   (parse_array == 19).astype(np.float32))

    r = 20
    agnostic = img.copy()
    agnostic_draw = ImageDraw.Draw(agnostic)

    length_a = np.linalg.norm(pose_data[5] - pose_data[2])
    length_b = np.linalg.norm(pose_data[12] - pose_data[9])
    point = (pose_data[9] + pose_data[12]) / 2
    pose_data[9] = point + (pose_data[9] - point) / length_b * length_a
    pose_data[12] = point + (pose_data[12] - point) / length_b * length_a

    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'gray', width=r * 10)
    for i in [2, 5]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx - r * 5, pointy - r * 5, pointx + r * 5, pointy + r * 5), 'gray', 'gray')
    for i in [3, 4, 6, 7]:
        if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or \
           (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
            continue
        agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'gray', width=r * 10)
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx - r * 5, pointy - r * 5, pointx + r * 5, pointy + r * 5), 'gray', 'gray')

    for i in [9, 12]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx - r * 3, pointy - r * 6, pointx + r * 3, pointy + r * 6), 'gray', 'gray')
    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 9]], 'gray', width=r * 6)
    agnostic_draw.line([tuple(pose_data[i]) for i in [5, 12]], 'gray', width=r * 6)
    agnostic_draw.line([tuple(pose_data[i]) for i in [9, 12]], 'gray', width=r * 12)
    agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], 'gray', 'gray')

    pointx, pointy = pose_data[1]
    agnostic_draw.rectangle((pointx - r * 7, pointy - r * 7, pointx + r * 7, pointy + r * 7), 'gray', 'gray')
    agnostic.paste(img, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
    agnostic.paste(img, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))
    return agnostic

# Labels for parsing map, from VITONDataset
_parse_labels = {
    0: ['background', [0, 10]], 1: ['hair', [1, 2]], 2: ['face', [4, 13]],
    3: ['upper', [5, 6, 7]], 4: ['bottom', [9, 12]], 5: ['left_arm', [14]],
    6: ['right_arm', [15]], 7: ['left_leg', [16]], 8: ['right_leg', [17]],
    9: ['left_shoe', [18]], 10: ['right_shoe', [19]], 11: ['socks', [8]],
    12: ['noise', [3, 11]]
}


def test_single_image(opt, seg, gmm, alias, device):
    print("Running in single image mode.")
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # --- Load Person Image and related data ---
    person_image_path = opt.person_img_path
    person_img_name = os.path.basename(person_image_path) # e.g. person.jpg
    person_base_name = person_img_name.split('.')[0] # e.g. person
    input_dir = os.path.dirname(person_image_path) # e.g. inputs/

    # Person image
    img = Image.open(person_image_path).convert('RGB')
    img = transforms.Resize((opt.load_height, opt.load_width), interpolation=Image.BICUBIC)(img) # PIL.Image

    # Pose image (RGB)
    # Expects: inputs/person_rendered.png
    pose_rgb_path = os.path.join(input_dir, f"{person_base_name}_rendered.png")
    if not os.path.exists(pose_rgb_path):
        raise FileNotFoundError(f"Required pose image not found: {pose_rgb_path}")
    pose_rgb = Image.open(pose_rgb_path).convert('RGB')
    pose_rgb = transforms.Resize((opt.load_height, opt.load_width), interpolation=Image.BICUBIC)(pose_rgb)
    pose_rgb_tensor = transform(pose_rgb).unsqueeze(0).to(device) # Add batch dim

    # Pose keypoints (JSON)
    # Expects: inputs/person_keypoints.json
    pose_json_path = os.path.join(input_dir, f"{person_base_name}_keypoints.json")
    if not os.path.exists(pose_json_path):
        raise FileNotFoundError(f"Required pose JSON not found: {pose_json_path}")
    with open(pose_json_path, 'r') as f:
        pose_label = json.load(f)
        pose_data = pose_label['people'][0]['pose_keypoints_2d']
        pose_data = np.array(pose_data).reshape((-1, 3))[:, :2]

    # Parsing image (segmentation map)
    # Expects: inputs/person_parse.png (simplified name)
    parse_path = os.path.join(input_dir, f"{person_base_name}_parse.png")
    if not os.path.exists(parse_path):
        # Fallback for original naming convention if simplified one is not found
        original_parse_dir = os.path.join(input_dir, "image-parse")
        original_parse_path = os.path.join(original_parse_dir, f"{person_base_name}.png")
        if os.path.exists(original_parse_path):
            parse_path = original_parse_path
            # Ensure the directory exists for image-parse if we are using it
            if not os.path.exists(original_parse_dir):
                 os.makedirs(original_parse_dir, exist_ok=True)
        else:
            raise FileNotFoundError(f"Required parsing image not found: {parse_path} or {original_parse_path}")

    parse_img = Image.open(parse_path) # Should be L mode or P mode
    parse_img = transforms.Resize((opt.load_width, opt.load_height), interpolation=Image.NEAREST)(parse_img)

    # Generate parse_agnostic
    parse_agnostic_pil = _get_parse_agnostic(parse_img.copy(), pose_data.copy(), opt.load_width, opt.load_height)
    parse_agnostic_np = np.array(parse_agnostic_pil)[None] # Add channel dim for LongTensor
    parse_agnostic_tensor = torch.from_numpy(parse_agnostic_np).long() # (1, H, W)

    parse_agnostic_map = torch.zeros(20, opt.load_height, opt.load_width, dtype=torch.float)
    parse_agnostic_map.scatter_(0, parse_agnostic_tensor, 1.0) # (20, H, W)
    
    # Remap to semantic_nc classes
    new_parse_agnostic_map = torch.zeros(opt.semantic_nc, opt.load_height, opt.load_width, dtype=torch.float)
    for i in range(len(_parse_labels)): # Use _parse_labels defined above
        for label_idx in _parse_labels[i][1]:
            new_parse_agnostic_map[i] += parse_agnostic_map[label_idx]
    parse_agnostic_final_tensor = new_parse_agnostic_map.unsqueeze(0).to(device) # (1, semantic_nc, H, W)

    # Generate img_agnostic
    img_agnostic_pil = _get_img_agnostic(img.copy(), parse_img.copy(), pose_data.copy(), opt.load_width, opt.load_height)
    img_agnostic_tensor = transform(img_agnostic_pil).unsqueeze(0).to(device) # (1, 3, H, W)
    
    # Original person image tensor
    img_tensor = transform(img).unsqueeze(0).to(device) # Not directly used by alias, but good to have if needed

    # --- Load Cloth Image and related data ---
    cloth_image_path = opt.cloth_img_path
    cloth_img_name = os.path.basename(cloth_image_path)
    cloth_base_name = cloth_img_name.split('.')[0]

    # Cloth image
    c_pil = Image.open(cloth_image_path).convert('RGB')
    c_pil = transforms.Resize((opt.load_width, opt.load_height), interpolation=Image.BICUBIC)(c_pil) # PIL.Image
    c_tensor = transform(c_pil).unsqueeze(0).to(device) # (1, 3, H, W)

    # Cloth mask
    # Expects: inputs/cloth-mask.jpg (or .png)
    cloth_mask_path_jpg = os.path.join(input_dir, f"{cloth_base_name}-mask.jpg")
    cloth_mask_path_png = os.path.join(input_dir, f"{cloth_base_name}-mask.png")
    cloth_mask_path = None
    if os.path.exists(cloth_mask_path_jpg):
        cloth_mask_path = cloth_mask_path_jpg
    elif os.path.exists(cloth_mask_path_png):
        cloth_mask_path = cloth_mask_path_png
    else:
        raise FileNotFoundError(f"Required cloth mask not found: {cloth_mask_path_jpg} or {cloth_mask_path_png}")

    cm_pil = Image.open(cloth_mask_path) # Should be L mode
    cm_pil = transforms.Resize((opt.load_width, opt.load_height), interpolation=Image.NEAREST)(cm_pil)
    cm_array = np.array(cm_pil)
    cm_array = (cm_array >= 128).astype(np.float32) # Binarize
    cm_tensor = torch.from_numpy(cm_array).unsqueeze(0).unsqueeze(0).to(device) # (1, 1, H, W)

    # --- Run Models ---
    with torch.no_grad():
        # Part 1. Segmentation generation
        parse_agnostic_down = F.interpolate(parse_agnostic_final_tensor, size=(256, 192), mode='bilinear')
        pose_down = F.interpolate(pose_rgb_tensor, size=(256, 192), mode='bilinear')
        c_masked_down = F.interpolate(c_tensor * cm_tensor, size=(256, 192), mode='bilinear')
        cm_down = F.interpolate(cm_tensor, size=(256, 192), mode='bilinear')
        
        # Ensure gen_noise gets a 4D tensor for size
        noise_size_tensor = cm_down 
        if noise_size_tensor.dim() == 3: # Should be (N,C,H,W)
             noise_size_tensor = noise_size_tensor.unsqueeze(0)

        seg_input = torch.cat((cm_down, c_masked_down, parse_agnostic_down, pose_down, gen_noise(noise_size_tensor.size()).to(device)), dim=1)

        up = nn.Upsample(size=(opt.load_height, opt.load_width), mode='bilinear').to(device)
        gauss = tgm.image.GaussianBlur((15, 15), (3, 3)).to(device)

        parse_pred_down = seg(seg_input)
        parse_pred = gauss(up(parse_pred_down))
        parse_pred = parse_pred.argmax(dim=1)[:, None] # (N, 1, H, W)

        parse_old = torch.zeros(parse_pred.size(0), 13, opt.load_height, opt.load_width, dtype=torch.float).to(device)
        parse_old.scatter_(1, parse_pred, 1.0)

        # Labels for model output parsing (different from _parse_labels for input)
        model_output_labels = {
            0: ['background', [0]], 1: ['paste', [2, 4, 7, 8, 9, 10, 11]],
            2: ['upper', [3]], 3: ['hair', [1]], 4: ['left_arm', [5]],
            5: ['right_arm', [6]], 6: ['noise', [12]]
        }
        parse_map_for_alias = torch.zeros(parse_pred.size(0), 7, opt.load_height, opt.load_width, dtype=torch.float).to(device)
        for j in range(len(model_output_labels)):
            for label_idx in model_output_labels[j][1]:
                parse_map_for_alias[:, j] += parse_old[:, label_idx]
        
        # Part 2. Clothes Deformation
        agnostic_gmm = F.interpolate(img_agnostic_tensor, size=(256, 192), mode='nearest')
        # Use parse_map_for_alias for GMM input if it represents the cloth region correctly
        # Original code uses parse[:, 2:3] which is 'upper' from its 'labels'
        # Here, parse_map_for_alias[:, 2:3] corresponds to 'upper' from model_output_labels
        parse_cloth_gmm = F.interpolate(parse_map_for_alias[:, 2:3], size=(256, 192), mode='nearest')
        pose_gmm = F.interpolate(pose_rgb_tensor, size=(256, 192), mode='nearest')
        c_gmm = F.interpolate(c_tensor, size=(256, 192), mode='nearest')
        gmm_input = torch.cat((parse_cloth_gmm, pose_gmm, agnostic_gmm), dim=1)

        _, warped_grid = gmm(gmm_input, c_gmm)
        warped_c = F.grid_sample(c_tensor, warped_grid, padding_mode='border', align_corners=False)
        warped_cm = F.grid_sample(cm_tensor, warped_grid, padding_mode='border', align_corners=False)

        # Part 3. Try-on synthesis
        misalign_mask = parse_map_for_alias[:, 2:3] - warped_cm
        misalign_mask[misalign_mask < 0.0] = 0.0
        parse_div = torch.cat((parse_map_for_alias, misalign_mask), dim=1) # Now 8 channels
        parse_div[:, 2:3] -= misalign_mask # parse_div's 3rd channel (upper body) is corrected

        # Alias input: (img_agnostic, pose, warped_c) = 3+3+3 = 9 channels
        # parse, parse_div, misalign_mask are other args
        output = alias(torch.cat((img_agnostic_tensor, pose_rgb_tensor, warped_c), dim=1), 
                       parse_map_for_alias, parse_div, misalign_mask)

        # Save the single output image
        output_filename = os.path.basename(opt.output_img_path)
        output_dir = os.path.dirname(opt.output_img_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving output to {opt.output_img_path}")
        save_images(output, [output_filename], output_dir) # Reusing save_images

    print("Single image processing finished.")


def test_dataset_mode(opt, seg, gmm, alias, device):
    # This is the original test function, renamed
    print("Running in dataset mode.")
    # Dynamically import VITONDataset and VITONDataLoader only if in dataset mode
    from datasets import VITONDataset, VITONDataLoader

    up = nn.Upsample(size=(opt.load_height, opt.load_width), mode='bilinear').to(device)
    gauss = GaussianBlur2d((15, 15), (3, 3)).to(device) # Use GaussianBlur2d

    test_dataset = VITONDataset(opt) # opt needs dataset_dir, dataset_mode, dataset_list etc.
    test_loader = VITONDataLoader(opt, test_dataset)

    # Ensure save directory for dataset mode exists
    dataset_save_dir = os.path.join(opt.save_dir, opt.name)
    if not os.path.exists(dataset_save_dir):
        os.makedirs(dataset_save_dir)

    with torch.no_grad():
        for i, inputs in enumerate(test_loader.data_loader):
            img_names = inputs['img_name']
            c_names = inputs['c_name']['unpaired']

            img_agnostic = inputs['img_agnostic'].to(device)
            parse_agnostic = inputs['parse_agnostic'].to(device) # This is new_parse_agnostic_map from dataset
            pose = inputs['pose'].to(device)
            c = inputs['cloth']['unpaired'].to(device)
            cm = inputs['cloth_mask']['unpaired'].to(device)

            # Part 1. Segmentation generation
            parse_agnostic_down = F.interpolate(parse_agnostic, size=(256, 192), mode='bilinear')
            pose_down = F.interpolate(pose, size=(256, 192), mode='bilinear')
            c_masked_down = F.interpolate(c * cm, size=(256, 192), mode='bilinear')
            cm_down = F.interpolate(cm, size=(256, 192), mode='bilinear')
            seg_input = torch.cat((cm_down, c_masked_down, parse_agnostic_down, pose_down, gen_noise(cm_down.size()).to(device)), dim=1)

            parse_pred_down = seg(seg_input)
            parse_pred = gauss(up(parse_pred_down))
            parse_pred = parse_pred.argmax(dim=1)[:, None]

            parse_old = torch.zeros(parse_pred.size(0), 13, opt.load_height, opt.load_width, dtype=torch.float).to(device)
            parse_old.scatter_(1, parse_pred, 1.0)

            # Labels for model output parsing
            model_output_labels = {
                0:  ['background',  [0]], 1:  ['paste', [2, 4, 7, 8, 9, 10, 11]],
                2:  ['upper',       [3]], 3:  ['hair',  [1]], 4:  ['left_arm', [5]],
                5:  ['right_arm',   [6]], 6:  ['noise', [12]]
            } # This is the 'labels' dict from original test function
            parse_map_for_alias = torch.zeros(parse_pred.size(0), 7, opt.load_height, opt.load_width, dtype=torch.float).to(device)
            for j_idx in range(len(model_output_labels)):
                for label in model_output_labels[j_idx][1]:
                    parse_map_for_alias[:, j_idx] += parse_old[:, label]

            # Part 2. Clothes Deformation
            agnostic_gmm = F.interpolate(img_agnostic, size=(256, 192), mode='nearest')
            parse_cloth_gmm = F.interpolate(parse_map_for_alias[:, 2:3], size=(256, 192), mode='nearest') # Using 'upper' from parse_map_for_alias
            pose_gmm = F.interpolate(pose, size=(256, 192), mode='nearest')
            c_gmm = F.interpolate(c, size=(256, 192), mode='nearest')
            gmm_input = torch.cat((parse_cloth_gmm, pose_gmm, agnostic_gmm), dim=1)

            _, warped_grid = gmm(gmm_input, c_gmm)
            warped_c = F.grid_sample(c, warped_grid, padding_mode='border', align_corners=False)
            warped_cm = F.grid_sample(cm, warped_grid, padding_mode='border', align_corners=False)

            # Part 3. Try-on synthesis
            misalign_mask = parse_map_for_alias[:, 2:3] - warped_cm
            misalign_mask[misalign_mask < 0.0] = 0.0
            parse_div = torch.cat((parse_map_for_alias, misalign_mask), dim=1)
            parse_div[:, 2:3] -= misalign_mask

            output = alias(torch.cat((img_agnostic, pose, warped_c), dim=1), parse_map_for_alias, parse_div, misalign_mask)

            unpaired_names = []
            for img_name_item, c_name_item in zip(img_names, c_names):
                # Extract base names and create a valid filename like person_cloth.png
                img_base = os.path.basename(img_name_item).split('.')[0]
                cloth_base = os.path.basename(c_name_item).split('.')[0]
                unpaired_names.append(f'{img_base}_{cloth_base}.png')
            
            save_images(output, unpaired_names, dataset_save_dir)

            if (i + 1) % opt.display_freq == 0:
                print("Dataset mode step: {}".format(i + 1))
    print("Dataset mode processing finished.")


def main():
    opt = get_opt()
    print("Options:", opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create SegGenerator, GMM, ALIASGenerator models
    # semantic_nc for SegGenerator output is opt.semantic_nc (e.g. 13)
    # semantic_nc for ALIASGenerator input parse map is 7 (derived from the 13 classes)
    
    # Seg model: input_nc = semantic_nc (from parse_agnostic) + 1 (cloth_mask) + 3 (cloth) + 3 (pose) + 1 (noise) ?
    # Original seg input: cm_down, c_masked_down, parse_agnostic_down, pose_down, noise
    # (1, 3, 13, 3, N) -> opt.semantic_nc + 8 if parse_agnostic is opt.semantic_nc
    # In VITONDataset, new_parse_agnostic_map is (opt.semantic_nc, H, W)
    # So, input to SegGenerator is opt.semantic_nc (from parse_agnostic) + 1 (cm) + 3 (c*cm) + 3 (pose) + N (noise_channels)
    # The SegGenerator in networks.py is defined as:
    # self.G = UnetGenerator(input_nc, output_nc, num_downs=5, ngf=64, norm_layer=norm_layer, use_dropout=True)
    # The `input_nc` for SegGenerator in original code is `opt.semantic_nc + 8`.
    # Let's assume `opt.semantic_nc` is the number of classes in `parse_agnostic_final_tensor` (13).
    # Then `cm_down` (1) + `c_masked_down` (3) + `parse_agnostic_down` (13) + `pose_down` (3) + `noise` (1, assuming gen_noise makes 1 channel)
    # 1 + 3 + 13 + 3 + 1 = 21.  Original is opt.semantic_nc (13) + 8 = 21. This matches.
    seg = SegGenerator(opt, input_nc=opt.semantic_nc + 8, output_nc=opt.semantic_nc) # opt.semantic_nc is 13 by default
    
    # GMM inputA_nc is 7 (parse_cloth_gmm from 7-channel parse map + pose_gmm + agnostic_gmm)
    # parse_cloth_gmm (1) + pose_gmm (3) + agnostic_gmm (3) = 7
    gmm = GMM(opt, inputA_nc=7, inputB_nc=3) # inputB_nc is cloth (3)
    
    # ALIAS semantic_nc is 7 (for the parse map it receives)
    # opt.semantic_nc is reset to 7 before ALIAS creation in original code
    # Input to ALIAS: main_input (img_agnostic, pose, warped_c) = 3+3+3=9
    # Other args: parse_map (7 channels), parse_div (8 channels), misalign_mask (1 channel)
    # So, ALIASGenerator(opt, input_nc=9) is correct for the main tensor.
    # The `semantic_nc` inside ALIASGenerator is used for its internal parsing stream.
    original_semantic_nc = opt.semantic_nc # Store 13
    opt.semantic_nc = 7 # For ALIAS internal use, consistent with its expected 7-channel parse map
    alias = ALIASGenerator(opt, input_nc=9)
    opt.semantic_nc = original_semantic_nc # Restore for other potential uses, though not strictly needed here

    # Ensure only the filename part of the checkpoint arguments is used for path joining
    seg_checkpoint_filename = os.path.basename(opt.seg_checkpoint)
    gmm_checkpoint_filename = os.path.basename(opt.gmm_checkpoint)
    alias_checkpoint_filename = os.path.basename(opt.alias_checkpoint)

    seg_checkpoint_path = os.path.join(opt.checkpoint_dir, seg_checkpoint_filename)
    gmm_checkpoint_path = os.path.join(opt.checkpoint_dir, gmm_checkpoint_filename)
    alias_checkpoint_path = os.path.join(opt.checkpoint_dir, alias_checkpoint_filename)

    load_checkpoint(seg_checkpoint_path, seg)
    load_checkpoint(gmm_checkpoint_path, gmm)
    load_checkpoint(alias_checkpoint_path, alias)

    seg.to(device).eval()
    gmm.to(device).eval()
    alias.to(device).eval()

    if opt.person_img_path and opt.cloth_img_path and opt.output_img_path:
        # API / Single Image Mode
        if not os.path.exists(opt.person_img_path):
            raise FileNotFoundError(f"Person image not found: {opt.person_img_path}")
        if not os.path.exists(opt.cloth_img_path):
            raise FileNotFoundError(f"Cloth image not found: {opt.cloth_img_path}")
        test_single_image(opt, seg, gmm, alias, device)
    elif opt.name and opt.dataset_list: # Check for dataset mode requirements
        # Dataset Mode
        test_dataset_mode(opt, seg, gmm, alias, device)
    else:
        print("Error: Insufficient arguments for either single image mode or dataset mode.")
        print("For single image: --person_img_path, --cloth_img_path, --output_img_path are required.")
        print("For dataset mode: --name, --dataset_list, --dataset_dir are typically required.")


if __name__ == '__main__':
    main()

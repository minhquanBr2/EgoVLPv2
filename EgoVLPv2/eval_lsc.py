import sys
sys.path.append('./')
import argparse
import os
import torch
import transformers
from csv import reader
from torchvision import transforms
from PIL import Image
import numpy as np
import shutil
import tqdm
from parse_config import ConfigParser
from utils import state_dict_data_parallel_fix
import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

import data_loader.data_loader as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model_epic_charades as module_arch

from trainer.trainer_charades import Multi_Trainer_dist_Charades, AllGather_multi


# print("Number of GPU: ", torch.cuda.device_count())
# print("GPU Name: ", torch.cuda.get_device_name())


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Define image preprocessing
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def print_config(config):
    if isinstance(config, dict):
        for key, value in config.items():
            if isinstance(value, dict):
                print(f"{key}: (Nested dict)")
                print_config(value)  # Recursively print nested dictionaries
            else:
                print(f"{key}: {value}")
    else:
        print(config)  # If config is not a dictionary, just print it

def prepare_text_embeds(cls_arr, tokenizer, model, device, config, args):
    texts = cls_arr
    tokenized_texts = tokenizer(
        texts, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    text_data = {
        "input_ids": tokenized_texts["input_ids"].to(device), 
        "attention_mask": tokenized_texts["attention_mask"].to(device)
    }

    with torch.no_grad():
        data = {
            'text': text_data,
            'video': None
        }
        z, t, ret = model(
            data, 
            allgather=None,
            n_gpu=config['n_gpu'],
            args=args,
            config=None,
            loss_dual=None,
            gpu="cuda",
            return_embeds=True
        )    
    text_embeds = ret['text_embeds'].cpu().detach()
    return text_embeds

def classify_image(image_path, model, tokenizer, text_embeds, device, config, args):
    """
    Classify a single image and return the predicted action class.
    
    Args:
        image_path (str): Path to the input image.
        model (torch.nn.Module): Loaded HierVL model.
        tokenizer (transformers.AutoTokenizer): Tokenizer for text input.
        cls_arr (list): List of action classes.
        device (torch.device): Device to run the model on.

    Returns:
        str: Predicted action class.
    """

    image = Image.open(image_path).convert("RGB")
    transformed_image = image_transform(image) 
    frames = transformed_image.unsqueeze(0).repeat(32, 1, 1, 1)
    video_data = frames.unsqueeze(0)  
    video_data = video_data.to(device)

    # Forward pass through the model
    model.eval()
    with torch.no_grad():
        data = {
            'text': None,
            'video': video_data
        }
        x, y, ret = model(
            data, 
            allgather=None,
            n_gpu=config['n_gpu'],
            args=args,
            config=None,
            loss_dual=None,
            gpu="cuda",
            return_embeds=True
        )

    # Compute similarity between text and image embeddings
    video_embeds = ret['video_embeds'].cpu().detach()
    sim_v2t = sim_matrix(video_embeds, text_embeds)
    print(f"Video Embeddings Shape: {video_embeds.shape}")
    print(f"Text Embeddings Shape: {text_embeds.shape}")
    print(f"Video-to-Text Similarity Shape: {sim_v2t.shape}")

    predicted_idx = torch.argmax(sim_v2t).item()
    return predicted_idx

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def save_image(output_dir, predicted_class, image_path):
    """
    Save the image to the output directory with the predicted action class as the filename.
    
    Args:
        output_dir (str): Path to the output directory.
        predicted_class (str): Predicted action class.
        image_path (str): Path to the input image.
    """
    dest_folder = os.path.join(output_dir, predicted_class)
    shutil.copy(image_path, dest_folder)
    print(f"Copied {image_path} to {dest_folder}")

def eval():
    args = argparse.ArgumentParser(description='PyTorch Action Recognition for Single Image')

    args.add_argument('-r', '--resume', default="./checkpoints/Charades-Ego_finetune.pth",
                      help='Path to latest checkpoint (default: None)')
    args.add_argument('-gpu', '--gpu', default=1, type=str,
                      help='Indices of GPUs to enable (default: all)')
    args.add_argument('-i', '--image', type=str,
                      help='Path to the input image')
    args.add_argument('--test_images_list_path', default=None, type=str,
                      help='Path to the list of test images.')    
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-c', '--config', default="./configs/eval/charades.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-s', '--sliding_window_stride', default=-1, type=int,
                      help='test time temporal augmentation, repeat samples with different start times.')
    args.add_argument('--save_feats', default=None,
                      help='path to store text & video feats, this is for saving embeddings if you want to do offline retrieval.')
    args.add_argument('--save_dir', default='./experiments', type=str)
    args.add_argument('--split', default='test', choices=['train', 'val', 'test'],
                      help='split to evaluate on.')
    args.add_argument('--batch_size', default=1, type=int,
                      help='size of batch')
    args.add_argument('--class_labels_file', default="D:/VSCode/THESIS/LaViLa/datasets/CharadesEgo/CharadesEgo/Charades_v1_classes.txt", type=str,)
    config = ConfigParser(args, test=True, eval_mode='charades')


    args = args.parse_args()
    args.world_size = 1
    os.environ["CUDA_VISIBLE_DEVICES"] =  "" + str(args.gpu)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load action classes
    cls_arr = []
    with open(args.class_labels_file, 'r') as charades:
        csv_reader = list(reader(charades))
    for line in csv_reader:
        cls_arr.append(line[0])

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config['arch']['args']['text_params']['model'])

    # Load model
    model = config.initialize('arch', module_arch)
    if config.resume is not None:
        checkpoint = torch.load(config.resume, map_location=device)
        state_dict = checkpoint['state_dict']
        new_state_dict = state_dict_data_parallel_fix(state_dict, model.state_dict())
        model.load_state_dict(new_state_dict, strict=False)
    else:
        print('Using random weights')
    model = model.to(device)

    # Prepare text embeds
    text_embeds = prepare_text_embeds(cls_arr, tokenizer, model, device, config, args)    

    # Classify the images
    test_images_list_path = args.test_images_list_path
    with open(test_images_list_path, "r") as f:
        test_files = [line.strip() for line in f.readlines()]
    for image_path in test_files:
        print(f"Classifying image: {image_path}")
        class_idx = classify_image(image_path, model, tokenizer, text_embeds, device, config, args)
        predicted_class = cls_arr[class_idx]
        print(f'Predicted Action Class: {predicted_class}')
        save_image(args.save_dir, predicted_class, image_path)
    # pathlib.PosixPath = temp

if __name__ == '__main__':
    eval()

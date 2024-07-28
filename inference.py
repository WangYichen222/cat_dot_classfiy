
import argparse
import torch
import torchvision
import torch.nn as nn
from mobilenetv3 import MobileNetV3_Small, MobileNetV3_Large
from datasets import build_transform
from PIL import Image

def pil_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class_to_idx ={
    'Abyssinian': 0, 
    'Bengal': 1, 
    'Birman': 2, 
    'Bombay': 3, 
    'British_Shorthair': 4, 
    'Egyptian_Mau': 5, 
    'Maine_Coon': 6, 
    'Persian': 7, 
    'Ragdoll': 8, 
    'Russian_Blue': 9, 
    'Siamese': 10, 
    'Sphynx': 11, 
    'american_bulldog': 12, 
    'american_pit_bull_terrier': 13, 
    'basset_hound': 14, 
    'beagle': 15, 
    'boxer': 16, 
    'chihuahua': 17, 
    'english_cocker_spaniel': 18, 
    'english_setter': 19, 
    'german_shorthaired': 20, 
    'great_pyrenees': 21, 
    'havanese': 22, 
    'japanese_chin': 23, 
    'keeshond': 24, 
    'leonberger': 25, 
    'miniature_pinscher': 26, 
    'newfoundland': 27, 
    'pomeranian': 28, 
    'pug': 29, 
    'saint_bernard': 30, 
    'samoyed': 31, 
    'scottish_terrier': 32, 
    'shiba_inu': 33, 
    'staffordshire_bull_terrier': 34, 
    'wheaten_terrier': 35, 
    'yorkshire_terrier': 36
}
idx_to_class = {}
for k,v in class_to_idx.items():
    idx_to_class[v] = k

index_to_subcategory = {
    1:'Abyssinian',
    2:'american_bulldog',
    3:'american_pit_bull_terrier',
    4:'basset_hound',
    5:'beagle',
    6:'Bengal',
    7:'Birman',
    8:'Bombay',
    9:'boxer',
    10:'British_Shorthair',
    11:'chihuahua',
    12:'Egyptian_Mau',
    13:'english_cocker_spaniel',
    14:'english_setter',
    15:'german_shorthaired',
    16:'great_pyrenees',
    17:'havanese',
    18:'japanese_chin',
    19:'keeshond',
    20:'leonberger',
    21:'Maine_Coon',
    22:'miniature_pinscher',
    23:'newfoundland',
    24:'Persian',
    25:'pomeranian',
    26:'pug',
    27:'Ragdoll',
    28:'Russian_Blue',
    29:'saint_bernard',
    30:'samoyed',
    31:'scottish_terrier',
    32:'shiba_inu',
    33:'Siamese',
    34:'Sphynx',
    35:'staffordshire_bull_terrier',
    36:'wheaten_terrier',
    37:'yorkshire_terrier',
}
species_to_category ={
    1:'Cat',
    2:'Dog',
    3:'Dog',
    4:'Dog',
    5:'Dog',
    6:'Cat',
    7:'Cat',
    8:'Cat',
    9:'Dog',
    10:'Cat',
    11:'Dog',
    12:'Cat',
    13:'Dog',
    14:'Dog',
    15:'Dog',
    16:'Dog',
    17:'Dog',
    18:'Dog',
    19:'Dog',
    20:'Dog',
    21:'Cat',
    22:'Dog',
    23:'Dog',
    24:'Cat',
    25:'Dog',
    26:'Dog',
    27:'Cat',
    28:'Cat',
    29:'Dog',
    30:'Dog',
    31:'Dog',
    32:'Dog',
    33:'Cat',
    34:'Cat',
    35:'Dog',
    36:'Dog',
    37:'Dog',
}
subcategory_to_index = {}
for k,v in index_to_subcategory.items():
    subcategory_to_index[v] = k

def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser('cat_dog inference script for image classification', add_help=False)
    parser.add_argument('--model', default='mobilenet_v3_small', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--img_path', default='', type=str, help='Path of img')
    parser.add_argument('--checkpoint', default='',help='resume from checkpoint')
    parser.add_argument('--device', default='cuda',help='device to use for training / testing')
    parser.add_argument('--input_size', default=224, type=int,help='image input size')
    parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True)
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--use_amp', type=str2bool, default=False, help="Use PyTorch's AMP (Automatic Mixed Precision) or not")
    args = parser.parse_args()
    return args

def main(args):
    device = torch.device(args.device)
    if args.model == "mobilenet_v3_small":
        model = MobileNetV3_Small(num_classes=37)
    elif args.model == "mobilenet_v3_large":
        model = MobileNetV3_Large(num_classes=37)
    elif args.model == 'resnet18':
        model = torchvision.models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 37)
    elif args.model == 'vit':
        # model = TinyViT(num_classes=37)
        model = tiny_vit_21m_224(pretrained=False)
    model.to(device)
    print("Model = %s" % str(model))
    
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

    img_transform = build_transform(is_train=False, args=args)
    img = pil_loader(args.img_path)
    input = img_transform(img)
    if args.use_amp:
        with torch.cuda.amp.autocast():    
            input = input.to(device, non_blocking=True)
            input = input.unsqueeze(0)   
            output = model(input)
    else:
        input = input.to(device, non_blocking=True)
        input = input.unsqueeze(0)   
        output = model(input)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.cpu().item()
    pred_subcategory = idx_to_class[pred]
    pred_ID = subcategory_to_index[pred_subcategory]
    pred_category = species_to_category[pred_ID]
    gt = '_'.join(args.img_path.split('/')[-1].split('_')[:-1])
    print('gt:  ID:{}, Category:{}, SubCategory:{}'.format(subcategory_to_index[gt], species_to_category[subcategory_to_index[gt]], gt))
    print('Pred:  ID:{}, Category:{}, SubCategory:{}'.format(pred_ID, pred_category, pred_subcategory))
    

if __name__ == '__main__':
    args = parse_args()
    main(args)
import torch
import yolo
from PIL import Image
from torchvision import transforms
import os

# COCO dataset, 80 classes
classes = ("rock", "paper", "scissors", "none")
ckpt_path = "E:\YOLO_V5\YOLOv5-PyTorch\checkpoint_200_512_640.pth"

# Create model with 4 classes (matches your checkpoint)
model = yolo.YOLOv5(num_classes=4, img_sizes=[512, 640], score_thresh=0.6)
# Load checkpoint
checkpoint = torch.load(ckpt_path,map_location="cpu")
model.load_state_dict(checkpoint["model"], strict=False)
model.eval()
print("Model loaded successfully with 4 classes.")
img = Image.open(r"E:\YOLO_V5\YOLOv5-PyTorch\dataset\rock_paper_coco_format\train\20220216_222059_jpg.rf.14313d32113212cae934a2fd1bb2d104.jpg").convert("RGB")
img = transforms.ToTensor()(img)
model.head.merge = False
images = [img]
results, losses = model(images)
yolo.show(images, results, classes, save="r000.jpg")



use_cuda = False
dataset = "coco" # currently only support VOC and COCO datasets
file_root = r"E:\YOLO_V5\YOLOv5-PyTorch\dataset\rock_paper_coco_format\valid"
ann_file = r"E:\YOLO_V5\YOLOv5-PyTorch\dataset\rock_paper_coco_format\annotations\instances_valid.json"
output_dir = "../yolov5s_val2017"

# create output directory
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
# choose device and show GPU information
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
cuda = device.type == "cuda"
if cuda: yolo.get_gpu_prop(show=True)
print("\ndevice: {}".format(device))
ds = yolo.datasets(dataset, file_root, ann_file, train=True)
dl = torch.utils.data.DataLoader(ds, shuffle=True, collate_fn=yolo.collate_wrapper, pin_memory=cuda)
# DataPrefetcher behaves like PyTorch's DataLoader, but it outputs CUDA tensors
d = yolo.DataPrefetcher(dl) if cuda else dl
model.to(device)


if ckpt_path:
    checkpoint = torch.load(ckpt_path, map_location=device)
    if "ema" in checkpoint:
        model.load_state_dict(checkpoint["ema"][0])
        print(checkpoint["eval_info"])
    else:
        model.load_state_dict(checkpoint)
    del checkpoint
    if cuda: torch.cuda.empty_cache()
for p in model.parameters():
    p.requires_grad_(False)

iters = 30

for i, data in enumerate(d):
    images = data.images
    targets = data.targets
    with torch.no_grad():
        results, losses = model(images)
    # images' saving names
    save = [os.path.join(output_dir, "{}.jpg".format(tgt["image_id"].item())) for tgt in targets]
    yolo.show(images, results, ds.classes, save)
    if i >= iters - 1:
        break
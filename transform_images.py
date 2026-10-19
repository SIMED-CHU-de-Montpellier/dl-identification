import os
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from transforms import CLAHETransform

input_root = "/Data/datasets/dicom/test"
output_root = "/Data/datasets/dicom/test-transform"

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)),
    transforms.Resize((224, 224)),
])

os.makedirs(output_root, exist_ok=True)

for cls in sorted(os.listdir(input_root)):
    input_cls_path = os.path.join(input_root, cls)
    output_cls_path = os.path.join(output_root, cls)

    if not os.path.isdir(input_cls_path):
        continue

    os.makedirs(output_cls_path, exist_ok=True)

    for img_name in tqdm(os.listdir(input_cls_path), desc=cls):
        input_path = os.path.join(input_cls_path, img_name)
        output_path = os.path.join(output_cls_path, img_name)

        img = Image.open(input_path).convert("RGB")
        img = transform(img)

        img.save(output_path)
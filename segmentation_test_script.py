import torchvision.transforms.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import urllib
import torch

from torchvision.utils import draw_segmentation_masks

# Based on pytorch tutorials

def plot_with_mask(input_image, all_masks, class_index, save_folder='results'):
    mask = (all_masks == class_index)
    convert_tensor = transforms.ToTensor()
    input_tensor = convert_tensor(input_image)
    input_tensor = (input_tensor * 255).type(torch.uint8)
    img = draw_segmentation_masks(input_tensor, masks=mask, alpha=0.7)
    # image_with_mask = torch.reshape(image_with_mask, (h, w, c))
    img = img.detach()
    img = F.to_pil_image(img)
    plt.imshow(np.asarray(img))
    plt.savefig(fname=f'{save_folder}/mask_class{class_index}.jpg')
    plt.clf()


def plot_only_segmentation_mask(all_masks):
    # create a color palette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    r = Image.fromarray(all_masks.byte().cpu().numpy()).resize(all_masks.size)
    # plot the semantic segmentation predictions of 21 classes in each color
    r.putpalette(colors)
    plt.imshow(r)
    plt.savefig(fname='results/testseg.jpg')
    plt.clf()


def main():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    # or any of these variants
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)
    model.eval()

    # Download an example image from the pytorch website
    url, filename = ("https://github.com/pytorch/hub/raw/master/images/deeplab1.png", "deeplab1.png")
    try: urllib.URLopener().retrieve(url, filename)
    except: urllib.request.urlretrieve(url, filename)

    # sample execution (requires torchvision)
    input_image = Image.open(filename)
    input_image = input_image.convert("RGB")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)  # torch.Size([3, 1026, 1282])
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    # torch.Size([1, 3, 1026, 1282])

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)  # encodes the class information (15, person, or 17, sheep, here)

    # ipdb> testoutput['aux'].shape
    # torch.Size([1, 21, 1026, 1282])
    # ipdb> testoutput['out'].shape
    # torch.Size([1, 21, 1026, 1282])

    # List of classes:
    # https://github.com/NVIDIA/DIGITS/blob/master/examples/semantic-segmentation/pascal-voc-classes.txt

    plot_only_segmentation_mask(output_predictions)
    plot_with_mask(input_image, output_predictions, 15)
    plot_with_mask(input_image, output_predictions, 17)


if __name__ == '__main__':
    main()
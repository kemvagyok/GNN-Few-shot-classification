from torchvision import transforms

def build_transform(img_size=28, grayscale=True):

    mean, std = ([0.5], [0.5]) if grayscale else (
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
    
    transform_list = []

    transform_list.append(transforms.Resize((img_size)))
    transform_list.append(transforms.CenterCrop((img_size)))

    if grayscale:
        transform_list.append(transforms.Grayscale(num_output_channels=1))

    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean, std))

    return transforms.Compose(transform_list)
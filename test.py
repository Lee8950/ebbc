import ebbc
import torch
import torchvision.transforms
import PIL.Image

net = ebbc.EBBC()
net.load_state_dict(torch.load("model_params.pth"))

transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.ToTensor(),])


with PIL.Image.open("q/000.jpg") as fp:
    #print(dir(fp))
    pic = transform(fp).unsqueeze(0).cuda()
    original_shape = (fp.height, fp.width)
    processed = net(pic)
    tensor_pic = processed.squeeze(0)
    tensor_pic = torchvision.transforms.Resize(original_shape)(tensor_pic)
    print(original_shape)
    pil_pic = torchvision.transforms.ToPILImage()(tensor_pic)
    pil_pic.show()
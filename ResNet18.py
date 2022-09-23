# %% [markdown]
# # Initialization

# %% [markdown]
# ## Import Libraries

# %%
from __future__ import print_function
from quantization_functions import train_loop
from quantization_functions import post_training_quant_model
from quantization_functions import quant_aware_resnet_model
import tqdm
from collections import namedtuple
import argparse
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import json

# Fast AI (PyTorch wrapper)
from fastai import *
from fastai.vision.all import *
import fastai
fastai.__version__

# %%

# %%
# make sure GPU is being used
# torch.cuda.current_device()
# torch.cuda.device(0)
# torch.cuda.get_device_name(0)

# %%
# Notebook auto reloads code. (Ref: http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython)
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ## Import Created Modules

# %%

# %% [markdown]
# ## Load Dataset

# %%
BATCH_SIZE = 128
TEST_BATCH_SIZE = 32
N_CLASS = 10

# %%
# Download Imagenette 320 pixel

path = untar_data(URLs.IMAGENETTE_320)

# %%
imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(112),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(*imagenet_stats, inplace=True)
])

test_tfms = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(112),
    transforms.ToTensor(),
    transforms.Normalize(*imagenet_stats)
])

# %%
# PyTorch datasets

trainset = datasets.ImageFolder(path/"train", train_tfms)
testset = datasets.ImageFolder(path/"val", test_tfms)

# PyTorch data loaders

train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=2)

# %% [markdown]
# # Resnet 18 Models

# %%
N_EPOCH = 6

# %%
SAVE_DIR = 'checkpoint/imagenette_resnet18'

# %% [markdown]
# ## Base model

base_model = torchvision.models.resnet18(pretrained=True)
base_model.fc = nn.Linear(base_model.fc.in_features,
                          N_CLASS)  # Change top layer

# ## Train Loop
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(base_model.parameters(),
                            1e-2, momentum=0.9, weight_decay=1e-5)
train_loop.train_model(
    train_dl=train_loader,
    val_dl=test_loader,
    model=base_model,
    optimizer=optimizer,
    criterion=criterion,
    clip_value=1e-2,
    epochs=N_EPOCH, save=f"{SAVE_DIR}/base_model"
)

base_model = 0
bit_8_model = 1
bit_7_model = 0
bit_6_model = 0
bit_5_model = 0
bit_4_model = 0
single_layer_quantization_model1 = 1
eval_only_4_model1 = 0
post_quant_training_mixed_model = 0
model_bits_file = f'quantization_functions/model_layers.json'
num_bit = 4
model_bits = json.load(open(model_bits_file, 'r'))
model_bits = model_bits['ResNet18_8_bit']

if base_model == 1:

    # %%
    base_model = torchvision.models.resnet18(pretrained=False)
    base_model.fc = nn.Linear(
        base_model.fc.in_features, N_CLASS)  # Change top layer

    base_model.load_state_dict(torch.load(
        f'{SAVE_DIR}/base_model/model_weights.pt'))

    # Validation accuracy
    train_loop.test_model(test_loader, base_model)
    model_weights_base = torch.load(
        f'{SAVE_DIR}/base_model/model_weights.pt', map_location='cpu')
    print(f'{SAVE_DIR}/base_model/model_weights.pt resnet base model weights')
    print(model_weights_base.keys())
    for k, v in base_model.state_dict().items():
        print(k)
# %% [markdown]
# ## Post Training Quantization

# %% [markdown]
# 8 bit quantization
if bit_8_model == 1:
    # # %%
    # Convert base model to a custom quantization layer with the trained weights
    c_base_model = quant_aware_resnet_model.CResnet18(num_class=10, q_num_bit=8, qat=False,
                                                      pretrained=f'{SAVE_DIR}/base_model/model_weights.pt')
    c_base_model.quantize(True)

    # Forward pass to have quantized weights
    print("testing loaded quantized resnet model")
    train_loop.test_model(train_loader, c_base_model)

    # %%
    # Convert to quantized model
    print("quantizing model")
    q_base_model = post_training_quant_model.QResnet18(num_class=10)
    q_base_model.convert_from(c_base_model)

    # %%
    # Validation accuracy
    print("testing 8-bit quantized model")
    train_loop.test_model(test_loader, q_base_model)

    # %%
    with open(f'{SAVE_DIR}/ptq8bit_model_weights.pt', 'wb') as f:
        torch.save(q_base_model.state_dict(), f)

# %% [markdown]
if bit_7_model == 1:
    # 7-bit quantization
    # %%
    # Convert base model to a custom quantization layer with the trained weights
    c_base_model = quant_aware_resnet_model.CResnet18(num_class=10, q_num_bit=7, qat=False,
                                                      pretrained=f'{SAVE_DIR}/base_model/model_weights.pt')
    c_base_model.quantize(True)

    # %%
    # Forward pass to have quantized weights
    train_loop.test_model(train_loader, c_base_model)

    # %%
    # Convert to quantized model
    q_base_model = post_training_quant_model.QResnet18(num_class=10)
    q_base_model.convert_from(c_base_model)

    # %%
    # Validation accuracy
    train_loop.test_model(test_loader, q_base_model)

    # %%
    with open(f'{SAVE_DIR}/ptq7bit_model_weights.pt', 'wb') as f:
        torch.save(q_base_model.state_dict(), f)

# # %% [markdown]
if bit_6_model == 1:
    # 6-bit quantization
    # %%
    # Convert base model to a custom quantization layer with the trained weights
    c_base_model = quant_aware_resnet_model.CResnet18(num_class=10, q_num_bit=6, qat=False,
                                                      pretrained=f'{SAVE_DIR}/base_model/model_weights.pt')
    c_base_model.quantize(True)

    # %%
    # Forward pass to have quantized weights
    print("testing loaded quantized resnet model")
    train_loop.test_model(train_loader, c_base_model)

    # %%
    # Convert to quantized model
    q_base_model = post_training_quant_model.QResnet18(num_class=10)
    q_base_model.convert_from(c_base_model)

    # %%
    # Validation accuracy
    print("testing 6-bit quantized model")
    train_loop.test_model(test_loader, q_base_model)

    # %%
    with open(f'{SAVE_DIR}/ptq6bit_model_weights.pt', 'wb') as f:
        torch.save(q_base_model.state_dict(), f)

# %% [markdown]
if bit_6_model == 1:

    # 5-bit quantization
    # %%
    # Convert base model to a custom quantization layer with the trained weights
    c_base_model = quant_aware_resnet_model.CResnet18(num_class=10, q_num_bit=5, qat=False,
                                                      pretrained=f'{SAVE_DIR}/base_model/model_weights.pt')
    c_base_model.quantize(True)

    # %%
    # Forward pass to have quantized weights
    print("testing loaded quantized resnet model")
    train_loop.test_model(train_loader, c_base_model)

    # %%
    # Convert to quantized model
    q_base_model = post_training_quant_model.QResnet18(num_class=10)
    q_base_model.convert_from(c_base_model)

    # %%
    # Validation accuracy
    print("testing 5-bit quantized model")
    train_loop.test_model(test_loader, q_base_model)

    # %%
    with open(f'{SAVE_DIR}/ptq5bit_model_weights.pt', 'wb') as f:
        torch.save(q_base_model.state_dict(), f)

# %% [markdown]
if bit_4_model == 1:
    # 4-bit quantization
    # %%
    # Convert base model to a custom quantization layer with the trained weights
    c_base_model = quant_aware_resnet_model.CResnet18(num_class=10, q_num_bit=4, qat=False,
                                                      pretrained=f'{SAVE_DIR}/base_model/model_weights.pt')
    c_base_model.quantize(True)

    # %%
    # Forward pass to have quantized weights
    print("testing loaded quantized resnet model")
    train_loop.test_model(train_loader, c_base_model)

    # %%
    # Convert to quantized model
    q_base_model = post_training_quant_model.QResnet18(num_class=10)
    q_base_model.convert_from(c_base_model)

    # %%
    # Validation accuracy
    print("testing 4-bit quantized model")
    train_loop.test_model(test_loader, q_base_model)

    # %%
    with open(f'{SAVE_DIR}/ptq4bit_model_weights.pt', 'wb') as f:
        torch.save(q_base_model.state_dict(), f)

# %% [markdown]
# ### mixed-bit quantization Kunal
if single_layer_quantization_model1 == 1:
    # %%
    # Convert base model to a custom quantization layer with the trained weights
    #parser = argparse.ArgumentParser(description='quantization sensitivity')
    # parser.add_argument("-l", "--layer",
    #                    type=int,
    #                    default="0",
    #                    help='name of example from example.py')
    #args = parser.parse_args()
    #four_bit_layer =args.layer

    for layer_no, layer_name in enumerate(model_bits.keys()):
        # if layer_no==four_bit_layer:
        time.sleep(1)
        model_bits[layer_name] = 4
        print(layer_name)
        c_base_model = quant_aware_resnet_model.CResnet18(
            num_class=10, q_num_bit=model_bits, qat=False, pretrained=f'{SAVE_DIR}/base_model/model_weights.pt')
        c_base_model.quantize(True)
        # print("c_base_model quantized")
        # print(c_base_model)
        # Forward pass to have quantized weights
        # print("testing loaded quantized resnet model")
        train_loop.test_model(train_loader, c_base_model)
        # %%
        # Convert to quantized model
        q_base_model = post_training_quant_model.QResnet18(num_class=10)
        q_base_model.convert_from(c_base_model)

        # %%
        # Validation accuracy
        train_loop.test_model(test_loader, q_base_model)
        del c_base_model
        del q_base_model
        torch.cuda.empty_cache()
        # print("loading weights from 8-bit quantized model")
        # model_weights = torch.load(
        #     f'{SAVE_DIR}/qat8bit/model_weights_quantized.pt', map_location='cpu')
        # q_base_model.load_state_dict(model_weights)
        # q_base_model.eval()
        # train_loop.test_model(test_loader, q_base_model)
        # for k, v in q_base_model.state_dict().items():
        #     print(k, v)
        # %%
        # with open(f'{SAVE_DIR}/ptq4bit_model_weights.pt', 'wb') as f:
        #    torch.save(q_base_model.state_dict(), f)

if eval_only_4_model1 == 1:
    # %% [markdown]
    # 4-bit quantization

    # %%
    # Create model with custom quantization layer from the start
    print(model_bits)
    c_base_model = quant_aware_resnet_model.CResnet18(
        num_class=10, q_num_bit=8, qat=True, pretrained=True)
    c_base_model.quantize(True)

    # %%
    # Training Loop
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        c_base_model.parameters(), 1e-3, momentum=0.9, weight_decay=1e-5)

    # train_loop.train_model(
    #     train_dl=train_loader,
    #     val_dl=test_loader,
    #     model=c_base_model,
    #     optimizer=optimizer,
    #     criterion=criterion,
    #     clip_value=1e-2,
    #     epochs=N_EPOCH, save=f"{SAVE_DIR}/qat4bit"
    # )

    # %%
    # Convert to quantized model
    q_base_model = post_training_quant_model.QResnet18(num_class=10)
    q_base_model.convert_from(c_base_model)

    # %%
    # Validation accuracy
    train_loop.test_model(test_loader, q_base_model)

    # %%
    # with open(f'{SAVE_DIR}/qat4bit/model_weights_quantized.pt', 'wb') as f:
    #     torch.save(q_base_model.state_dict(), f)


# # %% [markdown]
if post_quant_training_mixed_model == 1:
    # Create model with custom quantization layer from the start
    c_base_model = quant_aware_resnet_model.CResnet18(
        num_class=10, q_num_bit=model_bits, qat=True, pretrained=True)
    c_base_model.quantize(True)

    # %%
    # Training Loop
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        c_base_model.parameters(), 1e-3, momentum=0.9, weight_decay=1e-5)

    train_loop.train_model(
        train_dl=train_loader,
        val_dl=test_loader,
        model=c_base_model,
        optimizer=optimizer,
        criterion=criterion,
        clip_value=1e-2,
        epochs=N_EPOCH, save=f"{SAVE_DIR}/qat4bit"
    )

    # %%
    # Convert to quantized model
    q_base_model = post_training_quant_model.QResnet18(num_class=10)
    q_base_model.convert_from(c_base_model)

    # %%
    # Validation accuracy
    train_loop.test_model(test_loader, q_base_model)

    # %%
    with open(f'{SAVE_DIR}/qat4bit_modified/model_weights_quantized.pt', 'wb') as f:
        torch.save(q_base_model.state_dict(), f)

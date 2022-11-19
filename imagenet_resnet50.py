# %% [markdown]
# # Initialization

# %% [markdown]
# ## Import Libraries

# %%
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

# Fast AI (PyTorch wrapper)
from fastai import *
from fastai.vision.all import *
import fastai
fastai.__version__

# %%
from collections import namedtuple
import tqdm

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
from quantization_functions import quant_aware_resnet_model
from quantization_functions import post_training_quant_model
from quantization_functions import train_loop

# %% [markdown]
# ## Load Dataset

# %%
BATCH_SIZE = 128
TEST_BATCH_SIZE = 16
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
    transforms.Normalize(*imagenet_stats,inplace=True)
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

train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(testset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=2)

# %% [markdown]
# # Resnet 50 Models

# %%
N_EPOCH = 10

# %%
SAVE_DIR = 'checkpoint/imagenette_resnet50'

# %% [markdown]
# ## Base model

# %%
base_model = torchvision.models.resnet50(pretrained=True)
base_model.fc = nn.Linear(base_model.fc.in_features, N_CLASS) # Change top layer

# %%
### Train Loop
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(base_model.parameters(), 1e-2, momentum=0.9, weight_decay=1e-5)

train_loop.train_model(
    train_dl=train_loader,
    val_dl=test_loader,
    model=base_model,
    optimizer=optimizer,
    criterion=criterion,
    clip_value=1e-2,
    epochs=N_EPOCH, save=f"{SAVE_DIR}/base_model"
)

# %%
base_model = torchvision.models.resnet50(pretrained=False)
base_model.fc = nn.Linear(base_model.fc.in_features, N_CLASS) # Change top layer

base_model.load_state_dict(torch.load(f'{SAVE_DIR}/base_model/model_weights.pt'))

# Validation accuracy
train_loop.test_model(test_loader, base_model)

# %% [markdown]
# ## Post Training Quantization

# %% [markdown]
# ### 8 bit quantization

# %%
# Convert base model to a custom quantization layer with the trained weights
c_base_model = quant_aware_resnet_model.CResnet50(num_class=10, q_num_bit=8, qat=False,
                                                  pretrained=f'{SAVE_DIR}/base_model/model_weights.pt')
c_base_model.quantize(True)

# %%
# Forward pass to have quantized weights
train_loop.test_model(test_loader, c_base_model) # use test_loader to avoid out of memory

# %%
# Convert to quantized model
q_base_model = post_training_quant_model.QResnet50(num_class=10)
q_base_model.convert_from(c_base_model)

# %%
# Validation accuracy
train_loop.test_model(test_loader, q_base_model)

# %%
with open(f'{SAVE_DIR}/ptq8bit_model_weights.pt', 'wb') as f:
    torch.save(q_base_model.state_dict(), f)

# %% [markdown]
# ### 7-bit quantization

# %%
# Convert base model to a custom quantization layer with the trained weights
c_base_model = quant_aware_resnet_model.CResnet50(num_class=10, q_num_bit=7, qat=False,
                                                  pretrained=f'{SAVE_DIR}/base_model/model_weights.pt')
c_base_model.quantize(True)

# %%
# Forward pass to have quantized weights
train_loop.test_model(test_loader, c_base_model) # use test_loader to avoid out of memory

# %%
# Convert to quantized model
q_base_model = post_training_quant_model.QResnet50(num_class=10)
q_base_model.convert_from(c_base_model)

# %%
# Validation accuracy
train_loop.test_model(test_loader, q_base_model)

# %%
with open(f'{SAVE_DIR}/ptq7bit_model_weights.pt', 'wb') as f:
    torch.save(q_base_model.state_dict(), f)

# %% [markdown]
# ### 6-bit quantization

# %%
# Convert base model to a custom quantization layer with the trained weights
c_base_model = quant_aware_resnet_model.CResnet50(num_class=10, q_num_bit=6, qat=False,
                                                  pretrained=f'{SAVE_DIR}/base_model/model_weights.pt')
c_base_model.quantize(True)

# %%
# Forward pass to have quantized weights
train_loop.test_model(test_loader, c_base_model) # use test_loader to avoid out of memory

# %%
# Convert to quantized model
q_base_model = post_training_quant_model.QResnet50(num_class=10)
q_base_model.convert_from(c_base_model)

# %%
# Validation accuracy
train_loop.test_model(test_loader, q_base_model)

# %%
with open(f'{SAVE_DIR}/ptq6bit_model_weights.pt', 'wb') as f:
    torch.save(q_base_model.state_dict(), f)

# %% [markdown]
# ### 5-bit quantization

# %%
# Convert base model to a custom quantization layer with the trained weights
c_base_model = quant_aware_resnet_model.CResnet50(num_class=10, q_num_bit=5, qat=False,
                                                  pretrained=f'{SAVE_DIR}/base_model/model_weights.pt')
c_base_model.quantize(True)

# %%
# Forward pass to have quantized weights
train_loop.test_model(test_loader, c_base_model) # use test_loader to avoid out of memory

# %%
# Convert to quantized model
q_base_model = post_training_quant_model.QResnet50(num_class=10)
q_base_model.convert_from(c_base_model)

# %%
# Validation accuracy
train_loop.test_model(test_loader, q_base_model)

# %%
with open(f'{SAVE_DIR}/ptq5bit_model_weights.pt', 'wb') as f:
    torch.save(q_base_model.state_dict(), f)

# %% [markdown]
# ### 4-bit quantization

# %%
# Convert base model to a custom quantization layer with the trained weights
c_base_model = quant_aware_resnet_model.CResnet50(num_class=10, q_num_bit=4, qat=False,
                                                  pretrained=f'{SAVE_DIR}/base_model/model_weights.pt')
c_base_model.quantize(True)

# %%
# Forward pass to have quantized weights
train_loop.test_model(test_loader, c_base_model) # use test_loader to avoid out of memory

# %%
# Convert to quantized model
q_base_model = post_training_quant_model.QResnet50(num_class=10)
q_base_model.convert_from(c_base_model)

# %%
# Validation accuracy
train_loop.test_model(test_loader, q_base_model)

# %%
with open(f'{SAVE_DIR}/ptq4bit_model_weights.pt', 'wb') as f:
    torch.save(q_base_model.state_dict(), f)

# %% [markdown]
# ## Quantization Aware Training

# %% [markdown]
# ### 8-bit quantization

# %%
# Create model with custom quantization layer from the start
c_base_model = quant_aware_resnet_model.CResnet50(num_class=10, q_num_bit=8, qat=True, pretrained=True)
c_base_model.quantize(True)

# %%
# Training Loop
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(c_base_model.parameters(), 1e-3, momentum=0.9, weight_decay=1e-5)

train_loop.train_model(
    train_dl=train_loader,
    val_dl=test_loader,
    model=c_base_model,
    optimizer=optimizer,
    criterion=criterion,
    clip_value=1e-2,
    epochs=N_EPOCH, save=f"{SAVE_DIR}/qat8bit"
)

# %%
# Convert to quantized model
q_base_model = post_training_quant_model.QResnet50(num_class=10)
q_base_model.convert_from(c_base_model)

# %%
# Validation accuracy
train_loop.test_model(test_loader, q_base_model)

# %%
with open(f'{SAVE_DIR}/qat8bit/model_weights_quantized.pt', 'wb') as f:
    torch.save(q_base_model.state_dict(), f)

# %% [markdown]
# ### 7-bit quantization

# %%
# Create model with custom quantization layer from the start
c_base_model = quant_aware_resnet_model.CResnet50(num_class=10, q_num_bit=7, qat=True, pretrained=True)
c_base_model.quantize(True)

# %%
# Training Loop
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(c_base_model.parameters(), 1e-3, momentum=0.9, weight_decay=1e-5)

train_loop.train_model(
    train_dl=train_loader,
    val_dl=test_loader,
    model=c_base_model,
    optimizer=optimizer,
    criterion=criterion,
    clip_value=1e-2,
    epochs=N_EPOCH, save=f"{SAVE_DIR}/qat7bit"
)

# %%
# Convert to quantized model
q_base_model = post_training_quant_model.QResnet50(num_class=10)
q_base_model.convert_from(c_base_model)

# %%
# Validation accuracy
train_loop.test_model(test_loader, q_base_model)

# %%
with open(f'{SAVE_DIR}/qat7bit/model_weights_quantized.pt', 'wb') as f:
    torch.save(q_base_model.state_dict(), f)

# %% [markdown]
# ### 6-bit quantization

# %%
# Create model with custom quantization layer from the start
c_base_model = quant_aware_resnet_model.CResnet50(num_class=10, q_num_bit=6, qat=True, pretrained=True)
c_base_model.quantize(True)

# %%
# Training Loop
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(c_base_model.parameters(), 1e-2, momentum=0.9, weight_decay=1e-5)

train_loop.train_model(
    train_dl=train_loader,
    val_dl=test_loader,
    model=c_base_model,
    optimizer=optimizer,
    criterion=criterion,
    clip_value=1e-2,
    epochs=N_EPOCH, save=f"{SAVE_DIR}/qat6bit"
)

# %%
# Convert to quantized model
q_base_model = post_training_quant_model.QResnet50(num_class=10)
q_base_model.convert_from(c_base_model)

# %%
# Validation accuracy
train_loop.test_model(test_loader, q_base_model)

# %%
with open(f'{SAVE_DIR}/qat6bit/model_weights_quantized.pt', 'wb') as f:
    torch.save(q_base_model.state_dict(), f)

# %% [markdown]
# ### 5-bit quantization

# %%
# Create model with custom quantization layer from the start
c_base_model = quant_aware_resnet_model.CResnet50(num_class=10, q_num_bit=5, qat=True, pretrained=True)
c_base_model.quantize(True)

# %%
# Training Loop
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(c_base_model.parameters(), 1e-3, momentum=0.9, weight_decay=1e-5)

train_loop.train_model(
    train_dl=train_loader,
    val_dl=test_loader,
    model=c_base_model,
    optimizer=optimizer,
    criterion=criterion,
    clip_value=1e-2,
    epochs=N_EPOCH, save=f"{SAVE_DIR}/qat5bit"
)

# %%
# Convert to quantized model
q_base_model = post_training_quant_model.QResnet50(num_class=10)
q_base_model.convert_from(c_base_model)

# %%
# Validation accuracy
train_loop.test_model(test_loader, q_base_model)

# %%
with open(f'{SAVE_DIR}/qat5bit/model_weights_quantized.pt', 'wb') as f:
    torch.save(q_base_model.state_dict(), f)

# %% [markdown]
# ### 4-bit quantization

# %%
# Create model with custom quantization layer from the start
c_base_model = quant_aware_resnet_model.CResnet50(num_class=10, q_num_bit=4, qat=True, pretrained=True)
c_base_model.quantize(True)

# %%
# Training Loop
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(c_base_model.parameters(), 1e-3, momentum=0.9, weight_decay=1e-5)

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
q_base_model = post_training_quant_model.QResnet50(num_class=10)
q_base_model.convert_from(c_base_model)

# %%
# Validation accuracy
train_loop.test_model(test_loader, q_base_model)

# %%
with open(f'{SAVE_DIR}/qat4bit/model_weights_quantized.pt', 'wb') as f:
    torch.save(q_base_model.state_dict(), f)

# %%


# %%




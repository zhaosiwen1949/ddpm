from tqdm.auto import tqdm
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
from random import randrange

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import  Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torchvision.utils import save_image
from torch.optim import Adam

from model import Unet
from schedule import linear_beta_schedule
from data import image_size, channels, dataloader
from utils import extract, plot, num_to_group, save_gif

# 设置 schedule 相关参数
# timesteps 步数
timesteps = 300

# betas
betas = linear_beta_schedule(timesteps)

# alphas
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# 加噪公式
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

# 后验方差
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

# # 加载图片
# image = Image.open("./imgs/cat.jpg")

# # 转换图片
# image_size = 128
# transform = Compose(
#     [
#         Resize(image_size),
#         CenterCrop(image_size),
#         ToTensor(),
#         Lambda(lambda t: (t * 2.0) - 1.0),
#     ]
# )

# x_start = transform(image).unsqueeze(0)
# print(x_start.shape)


# reverse_transformers = Compose([
#     Lambda(lambda t: (t + 1) / 2),
#     Lambda(lambda t: t.permute(1,2,0)),
#     Lambda(lambda t: t * 255.0),
#     Lambda(lambda t: t.numpy().astype(np.uint8)),
#     ToPILImage(),
# ])
# image_resized = reverse_transformers(x_start.squeeze(0))
# image_resized.save("./imgs/cat_resized1111.jpg")

# 固定随机数种子
# torch.manual_seed(0)

# 前向加噪
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape) 
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, noise.shape)

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

# def get_noisy_image(x_start, t):
#     x_noisy = q_sample(x_start, t)

#     noisy_image = reverse_transformers(x_noisy.squeeze(0))

#     return noisy_image

# plot([get_noisy_image(x_start, torch.tensor([t])) for t in [0, 50, 100, 150, 199]])

# 损失函数
def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start, t, noise)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == "l1":
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == "l2":
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss

# 后向去噪过程
@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(
            posterior_variance, t, x.shape
        )
        noise = torch.randn(x.shape, device=x.device)
        return model_mean + posterior_variance_t * noise

@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device

    b = shape[0]

    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc="sampling loop time step", total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        # imgs.append(img.cpu().numpy())
        imgs.append(img.cpu())

    return imgs

@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))

# 训练代码
def train():
    result_folder = Path("./results")
    result_folder.mkdir(exist_ok=True)
    save_and_sample_every = 1000

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Unet(
        dim=image_size,
        channels=channels,
        dim_mults=(1, 2, 4)
    )
    model.to(device)

    optimizer = Adam(model.parameters(), lr=1e-3)

    epochs = 15
    step = 0

    for epoch in range(0, epochs):
        for index, batch in enumerate(dataloader):
            optimizer.zero_grad()

            batch_size = batch["pixel_values"].shape[0]
            batch = batch["pixel_values"].to(device)

            t = torch.randint(0, timesteps, (batch_size,), device=device, dtype=torch.long)

            loss = p_losses(model, batch, t, loss_type="huber")

            if step % 100 == 0:
                print("Loss:", loss.item())

            loss.backward()
            optimizer.step()

            if step != 0 and step % save_and_sample_every == 0:
                milestone = step // save_and_sample_every
                batches = num_to_group(batch_size, 4)
                all_images_list = list(map(lambda n: sample(model, image_size=image_size, batch_size=n, channels=channels)[-1], batches))
                all_images = torch.cat(all_images_list, dim=0)
                all_images = (all_images + 1) * 0.5
                save_image(all_images, str(result_folder / f'sample-epoch_{epoch}-step_{step}.png'), nrow = 6)

            step = step + 1

    torch.save(model.state_dict(), "./results/model.pt")

# 推理代码
def eval():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载模型
    model = Unet(
        dim=image_size,
        channels=channels,
        dim_mults=(1,2,4)
    ).to(device)
    model.load_state_dict(
        torch.load(
            "./results/model.pt",
            map_location=device,
            weights_only=True
        )
    )
    model.eval()

    # 采样
    batch_size = 128
    samples = sample(model, image_size=image_size, batch_size=batch_size, channels=channels)

    # 保存结果
    eval_folder = Path("./evals")
    eval_folder.mkdir(exist_ok=True)
    gif_name = datetime.now().strftime("%Y%m%d_%H%M%S_eval.gif")
    file_index = randrange(batch_size)
    save_gif(file_index, timesteps, samples, save_path=eval_folder / gif_name)

    
if __name__ == "__main__":
    # 解析参数
    parser = argparse.ArgumentParser(description="DDPM 脚本")

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "eval"],
        help="选择脚本模式：train、eval"
    )

    args = parser.parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "eval":
        eval()
    else:
        raise NotImplementedError()

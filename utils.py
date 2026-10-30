import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def num_to_group(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups

    if remainder > 0:
        arr.append(remainder)
    return arr

def plot(imgs, orig_img=None, row_title=None, **imshow_kwargs):
    print("Backend:", matplotlib.get_backend())

    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    with_orig = orig_img is not None

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(figsize=(12,8), nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [orig_img] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    plt.show()

def save_gif(index, timesteps, samples, save_path):
    fig = plt.figure()
    imgs = []
    for i in range(0, timesteps):
        img = plt.imshow(samples[i][index].squeeze(), cmap="gray", animated=True)
        imgs.append([img])

    animate = animation.ArtistAnimation(fig, imgs, interval=50, blit=True, repeat_delay=1000)
    animate.save(save_path)
    plt.show()
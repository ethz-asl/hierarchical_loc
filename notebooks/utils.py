import matplotlib.pyplot as plt
import cv2
import numpy as np


def plot_imgs(imgs, titles=None, cmap='brg', ylabel='', normalize=True, ax=None,
              r=(0, 1), dpi=100):
    n = len(imgs)
    if not isinstance(cmap, list):
        cmap = [cmap]*n
    if ax is None:
        _, ax = plt.subplots(1, n, figsize=(6*n, 6), dpi=dpi)
        if n == 1:
            ax = [ax]
    else:
        if not isinstance(ax, list):
            ax = [ax]
        assert len(ax) == len(imgs)
    for i in range(n):
        if len(imgs[i].shape) == 3:
            if imgs[i].shape[-1] == 3:
                imgs[i] = imgs[i][..., ::-1]  # BGR to RGB
            elif imgs[i].shape[-1] == 1:
                imgs[i] = imgs[i][..., 0]
        if len(imgs[i].shape) == 2 and cmap[i] == 'brg':
            cmap[i] = 'gray'
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmap[i]),
                     vmin=None if normalize else r[0],
                     vmax=None if normalize else r[1])
        if titles:
            ax[i].set_title(titles[i])
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
    ax[0].set_ylabel(ylabel)
    plt.tight_layout()


def draw_datches(img1, kp1, img2, kp2, matches, color=None, kp_radius=5,
                 thickness=2, margin=20):
    # Create frame
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]),
                     img1.shape[1]+img2.shape[1]+margin,
                     img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0],
                     img2.shape[0]),
                     img1.shape[1]+img2.shape[1]+margin)
    new_img = np.ones(new_shape, type(img1.flat[0]))*255

    # Place original images
    new_img[0:img1.shape[0], 0:img1.shape[1]] = img1
    new_img[0:img2.shape[0],
            img1.shape[1]+margin:img1.shape[1]+img2.shape[1]+margin] = img2

    # Draw lines between matches
    if color:
        c = color
    for m in matches:
        # Generate random color for RGB/BGR and grayscale images as needed.
        if not color:
            if len(img1.shape) == 3:
                c = np.random.randint(0, 256, 3)
            else:
                c = np.random.randint(0, 256)
            c = (int(c[0]), int(c[1]), int(c[2]))

        end1 = tuple(np.round(kp1[m.trainIdx].pt).astype(int))
        end2 = tuple(np.round(kp2[m.queryIdx].pt).astype(int)
                     + np.array([img1.shape[1]+margin, 0]))
        cv2.line(new_img, end1, end2, c, thickness, lineType=cv2.LINE_AA)
        cv2.circle(new_img, end1, kp_radius, c, thickness, lineType=cv2.LINE_AA)
        cv2.circle(new_img, end2, kp_radius, c, thickness, lineType=cv2.LINE_AA)
    return new_img

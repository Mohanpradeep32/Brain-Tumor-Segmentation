import os
import numpy as np
from modelzz import U_net
import nibabel as nib
import cv2
import matplotlib.pyplot as plt
from flask import Flask, render_template, request

SEGMENT_CLASSES = {
    0: 'NOT tumor',
    1: 'NECROTIC/CORE',  # or NON-ENHANCING tumor CORE
    2: 'EDEMA',
    3: 'ENHANCING'  # original 4 -> converted into 3 later
}
IMG_SIZE = 128
# there are 155 slices per volume
# to start at 5 and use 145 slices means we will skip the first 5 and last 5
VOLUME_SLICES = 100
VOLUME_START_AT = 22  # first slice of volume that we will include
cur_dirr = os.path.abspath(os.getcwd())


def predict_tumors(flair_path, ce_path, dropdown, start_slice=60):

    X = np.empty((VOLUME_SLICES, IMG_SIZE, IMG_SIZE, 2))

    flair = nib.load(flair_path).get_fdata()
    ce = nib.load(ce_path).get_fdata()

    for j in range(VOLUME_SLICES):
        X[j, :, :, 0] = cv2.resize(
            flair[:, :, j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
        X[j, :, :, 1] = cv2.resize(
            ce[:, :, j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))

    pred = unet.predict(X/np.max(X), verbose=1)

    plt.imshow(cv2.resize(flair[:, :, start_slice+VOLUME_START_AT],
                          (IMG_SIZE, IMG_SIZE)), cmap="gray",
               interpolation='none', alpha=0.7)

    if dropdown == SEGMENT_CLASSES[1]:
        edema = pred[:, :, :, 2]
        plt.imshow(edema[start_slice, :, :], cmap="OrRd",
                   interpolation='none', alpha=0.3)
        plt.title(f'{SEGMENT_CLASSES[1]} predicted')
        plt.axis(False)

    elif dropdown == SEGMENT_CLASSES[2]:
        core = pred[:, :, :, 1]
        plt.imshow(core[start_slice, :, :], cmap="OrRd",
                   interpolation='none', alpha=0.3)
        plt.title(f'{SEGMENT_CLASSES[2]} predicted')
        plt.axis(False)

    elif dropdown == SEGMENT_CLASSES[3]:
        enhancing = pred[:, :, :, 3]
        plt.imshow(enhancing[start_slice, :, :],
                   cmap="OrRd", interpolation='none', alpha=0.3)
        plt.title(f'{SEGMENT_CLASSES[3]} predicted')
        plt.axis(False)

    else:
        plt.imshow(pred[start_slice, :, :, 1:4], cmap="Reds",
                   interpolation='none', alpha=0.3)
        plt.title('all predicted classes')
        plt.axis(False)

    plt.tight_layout()
    plt.savefig(cur_dirr + '\\static\\out_img.jpg')
    # plt.show()


menuu = list(SEGMENT_CLASSES.values())[1:]

unet = U_net(input_shape=(IMG_SIZE, IMG_SIZE, 2), classes=len(SEGMENT_CLASSES))
unet.load_weights(r"C:\Users\MOHAN PRADEEP\Downloads\md_model_weights (1).h5")
print('\nready to launch...\n')


app = Flask(__name__)


@app.route('/')
def home():
    return render_template("home.html", menuu=menuu)


@app.route('/outpg', methods=["GET", "POST"])
def outtputs():

    flair_path = os.path.abspath(request.form.get("flair_path")[1:-1])
    ce_path = os.path.abspath(request.form.get("ce_path")[1:-1])
    select_tumor = (request.form.get("part_of_tumor"))

    if (not flair_path):
        return render_template("failure.html", message="missing flair path")
    if (not ce_path):
        return render_template("failure.html", message="missing ce path")
    if (select_tumor not in menuu and select_tumor != 'all_the_tumor'):
        return render_template("failure.html", message="invalid tumor part")

    predict_tumors(flair_path=flair_path,
                   ce_path=ce_path,
                   dropdown=select_tumor)

    flair = nib.load(flair_path).get_fdata()
    plt.imshow(cv2.resize(flair[:, :, 60+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)),
               cmap="gray", interpolation='none', alpha=1.0)
    plt.title('input image')
    plt.tight_layout()
    plt.savefig(cur_dirr + '\\static\\input.jpg')

    return render_template("success.html",
                           flair_path="input.png",
                           output_path="out_img.png")


if __name__ == '__main__':
    app.run(debug=True)

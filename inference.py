import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from network.ocrnet import MscaleOCR
from loss.optimizer import restore_net
from config import cfg, assert_and_infer_cfg
from patcher import split_in_chunks, merge_from_chunks


class Inference:
    def __init__(self, patch_size=1984, padding=32, model_path="./seg_weights/best_checkpoint_ep650.pth"):
        # set configs
        cfg.MODEL.N_SCALES = [0.25, 0.5, 1.0]
        assert_and_infer_cfg(None, train_mode=False)
        self.patch_size = patch_size
        self.padding = padding

        # load net
        self.net = MscaleOCR(7, criterion=None)
        self.net.cuda()
        self.net = torch.nn.DataParallel(self.net)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        restore_net(self.net, checkpoint)
        self.net.eval()

    def run(self, chunk):
        images = torch.from_numpy(chunk)
        images = images.permute(2, 0, 1)
        images = torch.unsqueeze(images, 0)

        inputs = {'images': images,
                  'gts': (images[:, :, :, 0] * 0).type(torch.long)}  # torch.zeros((1, h, w), dtype=torch.long)}
        inputs = {k: v.cuda() for k, v in inputs.items()}

        res = self.net(inputs)

        inputs = None
        del inputs

        return res

    def run_large(self, img, attention=True):

        chunks = split_in_chunks(img, self.patch_size, self.padding)

        pred, attn = {}, {}

        with torch.no_grad():
            for chunk in tqdm(chunks):
                # inference
                output_dict = infer.run(chunk)

                for key in output_dict.keys():
                    _pred = output_dict[key]

                    # get original size
                    _pred = torch.nn.functional.interpolate(_pred,
                                                            size=(self.patch_size + 2 * self.padding,
                                                                  self.patch_size + 2 * self.padding),
                                                            mode='bilinear',
                                                            align_corners=cfg.MODEL.ALIGN_CORNERS)
                    # attention
                    if _pred.shape[1] == 1:
                        if attention:
                            if not key in attn: attn[key] = []
                            attn[key].append(_pred[0, 0, ...].cpu().numpy())

                    # prediction
                    else:
                        if not key in pred: pred[key] = []
                        output_data = torch.nn.functional.softmax(_pred, dim=1).cpu().data
                        pred[key].append(output_data[0, ...].permute(1, 2, 0))

            # garbage collection
            output_dict = None
            _pred = None
            del output_dict
            del _pred
            torch.cuda.empty_cache()

        # attention
        if attention:
            for key in attn.keys():
                attn[key] = merge_from_chunks(attn[key], img.shape[0], img.shape[1], 1, self.patch_size, self.padding)

            attn_tmp = np.stack(list(attn.values()), axis=-1)
            from scipy.special import softmax
            attn_tmp = softmax(attn_tmp, axis=-1)
            for i, key in enumerate(attn.keys()):
                attn[key] = np.uint8(attn_tmp[..., i] * 255)
                # cv2.imwrite("./images/" + ff.replace(".", f"_{key}."), attn[key])
        else:
            attn = None

        # predictions
        for key in list(pred.keys()):
            pred_tmp = merge_from_chunks(pred[key], img.shape[0], img.shape[1], 7, self.patch_size, self.padding)
            if key == "pred":
                for i in range(7):
                    pred[f"{key}_{i}"] = np.uint8(pred_tmp[..., i] * 255)
                    # cv2.imwrite("./images/" + ff.replace(".", f"_{key}_{i}."), np.uint8(pred_tmp[..., i] * 255))
            pred_tmp = np.argmax(pred_tmp, axis=-1)
            pred_tmp = self.class2color_print2(pred_tmp)
            pred_tmp = cv2.cvtColor(pred_tmp, cv2.COLOR_RGB2BGR)
            pred[key] = pred_tmp
            # cv2.imwrite("./images/" + ff.replace(".", f"_{key}."), pred_tmp)

        return pred, attn

    @staticmethod
    def class2color_print2(pred):
        res = np.stack((pred,) * 3, axis=-1)
        res[np.where((res == [0, 0, 0]).all(axis=2))] = [255, 255, 255]
        res[np.where((res == [1, 1, 1]).all(axis=2))] = [0, 0, 0]
        res[np.where((res == [2, 2, 2]).all(axis=2))] = [228, 26, 28]
        res[np.where((res == [3, 3, 3]).all(axis=2))] = [255, 127, 0]
        res[np.where((res == [4, 4, 4]).all(axis=2))] = [55, 126, 184]
        res[np.where((res == [5, 5, 5]).all(axis=2))] = [77, 175, 74]
        res[np.where((res == [6, 6, 6]).all(axis=2))] = [152, 78, 163]
        return np.uint8(res)

    @staticmethod
    def normalize(img):
        # normalize image
        img = np.float32(img) / 255
        img = img - np.array([0.485, 0.456, 0.406])
        img = img / np.array([0.229, 0.224, 0.225])
        img = np.float32(img)
        return img


if __name__ == "__main__":

    infer = Inference(patch_size=1984, padding=32)

    source_dir = "/media/chrisbe/9812080e-2b1a-498a-81e8-99b092601af4/data/referenzobjekte/talbruecke_bruenn/2021_09_08_Jakob/bundler_segments/202008/images"
    target_dir = "/media/chrisbe/9812080e-2b1a-498a-81e8-99b092601af4/data/referenzobjekte/talbruecke_bruenn/2021_09_08_Jakob/bundler_segments/202008/predictions"

    files = os.listdir(source_dir)
    files.sort()

    for ff in files:

        if os.path.exists(os.path.join(target_dir, ff.replace(".", f"_{key}."))):
            continue

        img = cv2.imread(os.path.join(source_dir, ff), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = Inference.normalize(img)
        pred, attn = infer.run_large(img, attention=False)

        for key in pred.keys():
            cv2.imwrite(os.path.join(target_dir, ff.replace(".", f"_{key}.")), pred[key])

        if not attn is None:
            for key in attn.keys():
                print(key, type(attn[key]))
                cv2.imwrite("./images/" + ff.replace(".", f"_{key}."), attn[key])



import os
import matplotlib.pyplot as plt
import json
import pandas as pd
import seaborn as sns
from PIL import Image
import torch
# from lpips import LPIPS
import numpy as np
# from skimage.metrics import  structural_similarity as ssim
from loss_functions import psnr, lpips_ as lpips, ssim, computeMask

BASE_FOLDER = "D:/Unseen_non_DF/"
DATA_NAME = "data.csv"

# lpips_model = LPIPS(net='alex')  # Use 'alex' for AlexNet backbone
# lpips_model.eval()

def parse_directory_structure(base_path):
    # List to hold the extracted data
    data = []

    # Walk through the file structure
    for scene in os.listdir(base_path):
        scene_path = os.path.join(base_path, scene)
        if os.path.isdir(scene_path):
            for resolution in os.listdir(scene_path):
                if resolution == "imgs":
                    continue
                resolution_path = os.path.join(scene_path, resolution)
                if os.path.isdir(resolution_path):
                    decimation_ratio = resolution.split("_")[1]
                    for method in os.listdir(resolution_path):
                        method_path = os.path.join(resolution_path, method)
                        if os.path.isdir(method_path):
                            data_file = os.path.join(method_path, 'data.json')
                            if os.path.isfile(data_file):
                                with open(data_file, 'r') as f:
                                    try:
                                        content = json.load(f)
                                        data.append({
                                            'scene': scene,
                                            'resolution': resolution.split("_")[0] + "x" + str(int(int(resolution.split("_")[0])*3/4)),
                                            'decimation_ratio': decimation_ratio,
                                            'method': method,
                                            'nrPoints': content.get('nrPoints'),
                                            'avg_time': content.get('avg_time'),
                                            'ids': content.get('ids')
                                        })
                                    except json.JSONDecodeError as e:
                                        print(f"Error reading JSON file {data_file}: {e}")
    df = pd.DataFrame(data)
    return df.explode('ids').reset_index(drop=True)
    

def speedvspointsplot(df):
    df['nrPoints'] = pd.to_numeric(df['nrPoints'], errors='coerce')
    df['avg_time'] = pd.to_numeric(df['avg_time'], errors='coerce')
    df = df.dropna(subset=['nrPoints', 'avg_time'])

    df_ours = df[df['method'] == 'ours']
    df_pointersect = df[df['method'] == 'pointersect']

    sns.set_theme(style='whitegrid')

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=False)

    # Plot "ours" with lines connecting points of the same resolution
    sns.lineplot(
        data=df_ours,
        x='nrPoints',
        y=1 / df_ours['avg_time'],
        hue='resolution',
        style='resolution',
        markers=True,
        ax=axes[0],
        legend="full"
    )
    axes[0].set_title('Speed vs. Number of Points (Ours)', fontsize=14)
    axes[0].set_xlabel('Number of Points', fontsize=12)
    axes[0].set_ylabel('Speed (1 / Avg Time)', fontsize=12)
    axes[0].legend(title='Resolution')

    # Plot "pointersect" with lines connecting points of the same resolution
    sns.lineplot(
        data=df_pointersect,
        x='nrPoints',
        y=1 / df_pointersect['avg_time'],
        hue='resolution',
        style='resolution',
        markers=True,
        ax=axes[1],
        legend="full"
    )
    axes[1].set_title('Speed vs. Number of Points (Pointersect)', fontsize=14)
    axes[1].set_xlabel('Number of Points', fontsize=12)
    axes[1].set_ylabel('Speed (1 / Avg Time)', fontsize=12)
    axes[1].legend(title='Resolution')

    plt.tight_layout()
    plt.show()

def genTableForRes(df):
    res_df = df[df['decimation_ratio'] == 1]

    res_df['avg_time'] = pd.to_numeric(res_df['avg_time'], errors='coerce')
    res_df = res_df.dropna(subset=['nrPoints', 'avg_time'])
    res_df['speed'] = 1 / res_df['avg_time']

    summary = res_df.groupby(["method", "resolution"])[["PSNR", "LPIPS", "SSIM", "speed"]].mean().reset_index()
    summary.rename(columns={"method": "Method", "resolution": "Resolution", "PSNR": "PSNR (dB)", "LPIPS": "LPIPS", "SSIM": "SSIM", "speed": "Speed (fps)"}, inplace=True)

    summary["PSNR (dB)"] = summary["PSNR (dB)"].map("{:.2f}".format)
    summary["LPIPS"] = summary["LPIPS"].map("{:.2f}".format)
    summary["SSIM"] = summary["SSIM"].map("{:.2f}".format)
    summary["Speed (fps)"] = summary["Speed (fps)"].map("{:.4f}".format)

    return summary.to_latex(index=False,
                            label="tab:resolution_table",
                            caption="Comparison of different methods for different resolutions and decimation ratios",
                            column_format="llcccc",
                            escape=False)

def genTableForDec(df):
    dec_df = df[df['resolution'] == "1920x1440"]

    dec_df['avg_time'] = pd.to_numeric(dec_df['avg_time'], errors='coerce')
    dec_df = dec_df.dropna(subset=['nrPoints', 'avg_time'])
    dec_df['speed'] = 1 / dec_df['avg_time']

    summary = dec_df.groupby(["method", "decimation_ratio"])[["PSNR", "LPIPS", "SSIM", "speed"]].mean().reset_index()
    summary.rename(columns={"method": "Method", "decimation_ratio": "Decimation Ratio", "PSNR": "PSNR (dB)", "LPIPS": "LPIPS", "SSIM": "SSIM", "speed": "Speed (fps)"}, inplace=True)

    summary["PSNR (dB)"] = summary["PSNR (dB)"].map("{:.2f}".format)
    summary["LPIPS"] = summary["LPIPS"].map("{:.2f}".format)
    summary["SSIM"] = summary["SSIM"].map("{:.2f}".format)
    summary["Speed (fps)"] = summary["Speed (fps)"].map("{:.4f}".format)
    summary["Decimation Ratio"] = summary["Decimation Ratio"].map("{:.1f}".format)

    return summary.to_latex(index=False,
                            label="tab:decimation_table",
                            caption="Comparison of different methods for different resolutions and decimation ratios",
                            column_format="llcccc",
                            escape=False)


    
def concatImages(img_paths, padding = 10):
    images = [Image.open(img_path) for img_path in img_paths]

    if len(images) > 1:
        reference_height = images[1].height
    else:
        return images[0]

    if images[0].height != reference_height:
        images[0] = images[0].resize(
            (int(images[0].width * reference_height / images[0].height), reference_height),
            Image.Resampling.BICUBIC
        )

    images_resized = [images[0]] + images[1:]

    total_width = sum(img.width for img in images_resized) + padding * (len(images_resized) - 1)

    concatenated_image = Image.new("RGB", (total_width, reference_height), (255, 255, 255))

    x_offset = 0
    for img in images_resized:
        concatenated_image.paste(img, (x_offset, 0))
        x_offset += img.width + padding

    return concatenated_image

def lossCalc(img_path_gt, img_path_render):
    render = np.array(Image.open(img_path_render))

    gt_img = Image.open(img_path_gt)
    gt_img = np.array(gt_img.resize(
            (int(gt_img.width * render.shape[0] / gt_img.height), render.shape[0]),
            Image.Resampling.BICUBIC
        ))
    
    gt_tensor = torch.from_numpy(gt_img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    image_tensor = torch.from_numpy(render).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    gt_tensor = gt_tensor.to("cuda")
    image_tensor = image_tensor.to("cuda")

    mask = computeMask(gt_tensor)

    return ssim(image_tensor[0], gt_tensor[0], mask).cpu().item(), psnr(image_tensor[0], gt_tensor[0], mask).cpu().item(), lpips(image_tensor[0], gt_tensor[0], mask).cpu().item()

def processImages(df):
    unique_scenes = df['scene'].unique().tolist()
    unique_resolutions = df['resolution'].unique().tolist()
    unique_decimations = df['decimation_ratio'].unique().tolist()
    unique_methods = df['method'].unique().tolist()

    for scene in unique_scenes:
        for resolution in unique_resolutions:
            for decimation in unique_decimations:
                scene_df = df[(df['scene'] == scene) & (df['resolution'] == resolution) & (df['decimation_ratio'] == decimation) & (df['method'] == 'ours')]
                ids = scene_df['ids']
                for idx, id in enumerate(ids):
                    # if(os.path.exists(f"imgs/{scene}/{resolution}/{decimation}/{id}.png")):
                    #     continue

                    imgs = []

                    imgs.append(os.path.join(BASE_FOLDER, scene, "imgs", f"frame_{id:06d}.jpg")) # gt_img
                    imgs.append(os.path.join(BASE_FOLDER, scene, f"{resolution.split('x')[0]}_{decimation}", "ours", "batch_0", f"input_{idx}.png")) # ours_input
                    imgs.append(os.path.join(BASE_FOLDER, scene, f"{resolution.split('x')[0]}_{decimation}", "ours", "batch_0", f"df_{idx}.png")) # ours_depthfiltered
                    imgs.append(os.path.join(BASE_FOLDER, scene, f"{resolution.split('x')[0]}_{decimation}", "ours", "batch_0", f"rgb_{idx}.png")) # ours_img

                    for method in unique_methods:
                        if method != "ours":
                            imgs.append(os.path.join(BASE_FOLDER, scene, f"{resolution.split('x')[0]}_{decimation}", method, "batch_0", f"rgb_{idx}.png"))


                    # loss calculation
                    for method in unique_methods:
                        render_path = os.path.join(BASE_FOLDER, scene, f"{resolution.split('x')[0]}_{decimation}", method, "batch_0", f"rgb_{idx}.png")
                        gt_path = imgs[0]

                        ssim, psnr, lpips = lossCalc(gt_path, render_path)
                        print(ssim, psnr, lpips)

                        row_index = df[(df['scene'] == scene) & (df['resolution'] == resolution) & (df['decimation_ratio'] == decimation) & (df['method'] == method) & (df['ids'] == id)].index[0]
                        df.at[row_index, "SSIM"] = ssim
                        df.at[row_index, "PSNR"] = psnr
                        df.at[row_index, "LPIPS"] = lpips


                    concatenated_image = concatImages(imgs)
                    os.makedirs(f"imgs_ndf/{scene}/{resolution}/{decimation}", exist_ok=True)
                    concatenated_image.save(f"imgs_ndf/{scene}/{resolution}/{decimation}/{id}.png")



if __name__ == '__main__':
    if  not os.path.exists(DATA_NAME):
        df = parse_directory_structure(BASE_FOLDER)
        processImages(df)
        df.to_csv(DATA_NAME, index=False)
    else:
        df = pd.read_csv(DATA_NAME)
    print(genTableForRes(df))
    print(genTableForDec(df))
    print(df[df["decimation_ratio"] == 1.0]["nrPoints"].min())
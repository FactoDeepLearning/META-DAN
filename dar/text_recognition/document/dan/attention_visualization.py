import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from dar.generic.utils_visualization import add_custom_color_maps


def generate_dan_visualization(input_img, attention_weights, preds, output_path, dataset_name="READ"):
    add_custom_color_maps()
    matplotlib.use("Agg")

    convert_special_token_read = {
        "ⓑ": "<body>",
        "Ⓑ": "</body>",
        "ⓐ": "<annotation>",
        "Ⓐ": "</annotation>",
        "ⓟ": "<page>",
        "Ⓟ": "</page>",
        "ⓝ": "<number>",
        "Ⓝ": "</number>",
        "ⓢ": "<section>",
        "Ⓢ": "</section>"
    }

    convert_special_token_rimes = {
        "ⓑ": "<body>",
        "Ⓑ": "</body>",
        "ⓞ": "<opening>",
        "Ⓞ": "</opening>",
        "ⓟ": "<PS-PJ>",
        "Ⓟ": "</PS-PJ>",
        "ⓡ": "<recipient-coordinates>",
        "Ⓡ": "</recipient-coordinates>",
        "ⓢ": "<sender-coordinates>",
        "Ⓢ": "</sender-coordinates>",
        "ⓦ": "<place-location>",
        "Ⓦ": "</place-location>",
        "ⓨ": "<subject>",
        "Ⓨ": "</subject>"
    }

    begin_tokens = "ⓑⓐⓟⓝⓢ" if dataset_name == "READ" else "ⓑⓞⓟⓡⓢⓦⓨ"
    end_tokens = "ⒷⒶⓅⓃⓈ" if dataset_name == "READ" else "ⒷⓄⓅⓇⓈⓌⓎ"
    convert_special_token = convert_special_token_read if dataset_name == "READ" else convert_special_token_rimes

    def format_text(text):
        tab = "    "
        res = list()
        num_tab = 0
        for ind in range(len(text)):
            char = text[ind]
            if char in begin_tokens and char != "":
                res.append("\n")
                for _ in range(num_tab):
                    res.append(tab)
                res.append(convert_special_token[char])
                res.append("\n")
                num_tab += 1

            elif char in end_tokens and char != "":
                res.append("\n")
                num_tab -= 1
                for _ in range(num_tab):
                    res.append(tab)
                res.append(convert_special_token[char])
                res.append("\n")

            else:
                if ind > 0 and text[ind-1] in begin_tokens:
                    for _ in range(num_tab):
                        res.append(tab)
                res.append(char)
            if char == "\n":
                for _ in range(num_tab):
                    res.append(tab)
        return "".join(res).replace("\n\n", "\n")

    if isinstance(attention_weights, list):
        attention_weights = np.stack(attention_weights, axis=0)

    preds = list(preds)
    preds.append("")
    preds.insert(0, "")

    text_size = 8
    anchor_x, anchor_y = (0.05, 0.95)

    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(12, 10)
    plt.axis('off')
    plt.suptitle("Denis Coquenet, Clément Chatelain, Thierry Paquet, \"DAN: a Segmentation-free Document Attention Network for Handwritten Document Recognition\"")

    zero_weights = np.zeros((1, attention_weights.shape[1], attention_weights.shape[2]))
    attention_weights = np.concatenate([zero_weights, attention_weights, zero_weights], axis=0)

    ax_img = plt.subplot(gs[:10, :4])
    ax_img.get_yaxis().set_visible(False)
    ax_img.get_xaxis().set_visible(False)
    ax_img.title.set_text('Input document')

    ax_cbar = plt.subplot(gs[10:11, :4])
    ax_cbar.get_yaxis().set_visible(False)
    ax_cbar.get_xaxis().set_visible(False)
    ax_cbar.axis('off')

    ax_text = plt.subplot(gs[11:, :4])
    ax_text.get_yaxis().set_visible(False)
    ax_text.get_xaxis().set_visible(False)
    ax_text.axis('off')

    ax_pred = plt.subplot(gs[:, 5:])
    ax_pred.get_yaxis().set_visible(False)
    ax_pred.get_xaxis().set_visible(False)
    ax_pred.title.set_text('Prediction')

    progressbar = tqdm(total=len(preds), mininterval=0, miniters=0)
    texts = {
        "iter": ax_pred.text(0.5, 0.25, "", size=12, ha='center'),
        "already_pred": ax_pred.text(anchor_x, anchor_y, "", size=text_size, va='top'),
        "already_pred2": ax_pred.text(0.51, 0.99, "", size=text_size, va='top'),
        "new_pred": ax_pred.text(0.5, 0.2, "", size=15, ha='center', va='center', weight="bold"),
    }
    img_input = ax_img.imshow(input_img)
    vmax = 0.2
    norm = Normalize(vmin=0, vmax=vmax)
    img_weight = ax_img.imshow(attention_weights[0], cmap="red_alpha", alpha=1, extent=img_input.get_extent(), norm=norm)
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap="red_alpha_bar"), ax=ax_cbar, orientation="horizontal", drawedges=False, fraction=1, shrink=0.5, anchor=(0.5, 1), ticks=[0, vmax])
    colorbar.set_label("Attention weights")
    colorbar.ax.set_xticklabels(['0', '{:.1f}+'.format(vmax)])
    plt.subplots_adjust(wspace=0.1, hspace=1, left=0.05, right=0.95, top=0.9, bottom=0.05)

    def update(i):
        progressbar.update(1)
        img_weight.set_data(attention_weights[i])
        already_predicted_text = "".join(preds[:i]) if i > 0 else ""
        pred_list = [already_predicted_text]
        pred_list = [format_text(p) for p in pred_list]
        texts["already_pred"].set_text(pred_list[0] if len(pred_list) > 0 else "")
        texts["already_pred2"].set_text(pred_list[1] if len(pred_list) > 1 else "")
        new_pred = convert_special_token[preds[i]] if preds[i] in convert_special_token.keys() else "<space>" if preds[i] == " " else "<linebreak>" if preds[i] == "\n" else preds[i]
        texts["new_pred"].set_text(new_pred)
        texts["iter"].set_text("Iteration: {}".format(min(i, len(preds) - 2)))
        return img_weight

    if output_path.split(".")[-1] == "mp4":
        anim = animation.FuncAnimation(fig, update, frames=np.arange(0, len(preds)), interval=1000, repeat=False) #, cache_frame_data=False
        anim.save(output_path, writer='ffmpeg', fps=1, dpi=150)
    else:
        anim = animation.FuncAnimation(fig, update, frames=np.arange(0, len(preds)), interval=1000, repeat=True)
        anim.save(output_path, writer='imagemagick', fps=1, dpi=150)

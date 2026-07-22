import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from dar.generic.utils_visualization import add_custom_color_maps


def generate_faster_dan_attention(input_img, line_att_weights, line_preds, name, dataset_name="READ"):
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
    all_tokens = begin_tokens + end_tokens
    convert_special_token = convert_special_token_read if dataset_name == "READ" else convert_special_token_rimes

    def format_gt(gt, indices=[]):
        bold_indices = list()
        tab = "    "
        res = list()
        num_tab = 0
        for ind in range(len(gt)):
            char = gt[ind]
            if char in begin_tokens and char != "":
                res.append("\n")
                for _ in range(num_tab):
                    res.append(tab)
                res.append(convert_special_token[char])
                if ind in indices:
                    bold_indices.append(len(res)-1)
                res.append("\n")
                num_tab += 1

            elif char in end_tokens and char != "":
                res.append("\n")
                num_tab -= 1
                for _ in range(num_tab):
                    res.append(tab)
                res.append(convert_special_token[char])
                if ind in indices:
                    bold_indices.append(len(res)-1)
                res.append("\n")

            else:
                if ind > 0 and gt[ind-1] in begin_tokens:
                    for _ in range(num_tab):
                        res.append(tab)
                res.append(char)
                if ind in indices:
                    bold_indices.append(len(res)-1)
            if char == "\n":
                for _ in range(num_tab):
                    res.append(tab)
        for ind in bold_indices:
            res[ind] = r"$\bf{"+res[ind].replace("\"", "''")+"}$" if res[ind] not in [" ", "\n"] else res[ind]
        return "".join(res).replace("\n\n", "\n")

    text_size = 8
    anchor_x, anchor_y = (0.05, 0.95)

    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(12, 10)
    plt.axis('off')
    plt.suptitle("Denis Coquenet, Clément Chatelain, Thierry Paquet, \"Faster DAN: Multi-target Queries with Document Positional Encoding for End-to-end Handwritten Document Recognition\"")

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

    ax_pred = plt.subplot(gs[:, 4:])
    ax_pred.get_yaxis().set_visible(False)
    ax_pred.get_xaxis().set_visible(False)
    ax_pred.title.set_text('Prediction')

    num_steps_first_pass = len(line_preds)
    num_steps = num_steps_first_pass + max([len(p) for p in line_preds])

    progressbar = tqdm(total=num_steps, mininterval=0, miniters=0)
    texts = {
        "iter": ax_pred.text(0.75, 0.975, "", size=12, ha='center'),
        "already_pred": ax_pred.text(anchor_x, anchor_y, "", size=text_size, va='top'),
        "already_pred2": ax_pred.text(0.51, 0.99, "", size=text_size, va='top'),
        "pass": ax_pred.text(0.25, 0.98, r"First pass", size=12, ha='center', va='center'),
    }
    img_input = ax_img.imshow(input_img)
    vmax = 0.2
    norm = Normalize(vmin=0, vmax=vmax)

    img_weight = ax_img.imshow(np.zeros(line_att_weights[0].shape[1:]), cmap="red_alpha", alpha=1, extent=img_input.get_extent(), norm=norm)
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap="red_alpha_bar"), ax=ax_cbar, orientation="horizontal", drawedges=False, fraction=1, shrink=0.5, anchor=(0.5, 1), ticks=[0, vmax])
    colorbar.set_label("Attention weights")
    colorbar.ax.set_xticklabels(['0', '{:.1f}+'.format(vmax)])
    plt.subplots_adjust(wspace=0.1, hspace=1, left=0.05, right=0.95, top=0.9, bottom=0.05)

    def update(i):
        progressbar.update(1)
        if i == 0:
            already_predicted_text = ""
            indices = None
        elif i <= num_steps_first_pass:
            predicted_lines = [l[0] for l in line_preds[:i]]
            predicted_lines = [predicted_lines[j] + "\n" if (j < len(predicted_lines) - 1 and predicted_lines[j] not in all_tokens and predicted_lines[j+1] not in all_tokens) else predicted_lines[j] for j in range(len(predicted_lines))]
            already_predicted_text = "".join(predicted_lines)
            img_weight.set_data(line_att_weights[i-1][0])
            indices = [len(already_predicted_text)-1, ]
        else:
            num_char_per_line = i - num_steps_first_pass + 1
            predicted_lines = [l[:num_char_per_line] for l in line_preds[:i]]
            predicted_lines = [predicted_lines[j] + "\n" if (j < len(predicted_lines) - 1 and predicted_lines[j] not in all_tokens and predicted_lines[j + 1] not in all_tokens) else predicted_lines[j] for j in range(len(predicted_lines))]
            index_in_lines = np.concatenate([np.arange(0, len(l)) for l in predicted_lines], axis=0)
            already_predicted_text = "".join(predicted_lines)

            img_weight.set_data(np.sum([w[num_char_per_line-1] for w in line_att_weights if w.shape[0] >= num_char_per_line], axis=0))

            indices = list(np.argwhere(index_in_lines==num_char_per_line-1)[:, 0])

        pred_list = [already_predicted_text]
        pred_list = [format_gt(p, indices) for p in pred_list]
        texts["already_pred"].set_text(pred_list[0] if len(pred_list) > 0 else "")
        texts["already_pred2"].set_text(pred_list[1] if len(pred_list) > 1 else "")
        if i == num_steps_first_pass + 1:
            texts["pass"].set_text(r"Second pass")
        texts["iter"].set_text("Iteration: {}".format(min(i, num_steps - 2)))
        return img_weight

    if name.split(".")[-1] == "mp4":
        anim = animation.FuncAnimation(fig, update, frames=np.arange(0, num_steps), interval=1000, repeat=False) #, cache_frame_data=False
        anim.save(name, writer='ffmpeg', fps=1, dpi=150)
    else:
        anim = animation.FuncAnimation(fig, update, frames=np.arange(0, num_steps), interval=1000, repeat=True)
        anim.save(name, writer='imagemagick', fps=1, dpi=150)

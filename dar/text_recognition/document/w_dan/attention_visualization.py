import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from dar.generic.utils_visualization import add_custom_color_maps, format_display_tokens


def generate_w_dan_visualization(input_img, attention_weights, preds, output_path):
    add_custom_color_maps()
    matplotlib.use("Agg")
    color_maps = ["red_alpha", "blue_alpha", "green_alpha", "yellow_alpha", "cyan_alpha"]
    
    num_iter = attention_weights.shape[0]
    token_win = attention_weights.shape[1]
    text_size = 8
    anchor_x, anchor_y = (0.05, 0.95)

    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(10+token_win, 10)
    plt.axis('off')
    plt.suptitle("Denis Coquenet, \"W-DAN\"")

    zero_weights = np.zeros((1, token_win, attention_weights.shape[2], attention_weights.shape[3]))
    attention_weights = np.concatenate([zero_weights, attention_weights, zero_weights], axis=0)

    ax_img = plt.subplot(gs[:8, :4])
    ax_img.get_yaxis().set_visible(False)
    ax_img.get_xaxis().set_visible(False)
    ax_img.title.set_text('Input document')

    ax_cbars = list()
    for i in range(token_win):
        ax_cbars.append(plt.subplot(gs[8+i:8+i+1, :4]))
        ax_cbars[-1].get_yaxis().set_visible(False)
        ax_cbars[-1].get_xaxis().set_visible(False)
        ax_cbars[-1].axis('off')

    ax_pred = plt.subplot(gs[:, 5:])
    ax_pred.get_yaxis().set_visible(False)
    ax_pred.get_xaxis().set_visible(False)
    ax_pred.title.set_text('Prediction')

    progressbar = tqdm(total=num_iter, mininterval=0, miniters=0)
    texts = {
        "iter": ax_pred.text(0.5, 0.25, "", size=12, ha='center'),
        "already_pred": ax_pred.text(anchor_x, anchor_y, "", size=text_size, va='top'),
        "already_pred2": ax_pred.text(0.51, 0.99, "", size=text_size, va='top'),
        "new_pred": ax_pred.text(0.5, 0.2, "", size=15, ha='center', va='center', weight="bold"),
    }
    img_input = ax_img.imshow(input_img)
    vmax = 0.2
    norm = Normalize(vmin=0, vmax=vmax)
    img_weights = list()
    for i in range(token_win):
        img_weights.append(ax_img.imshow(attention_weights[0, i], cmap=color_maps[i], alpha=1, extent=img_input.get_extent(), norm=norm))

        cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=f"{color_maps[i]}_bar"), ax=ax_cbars[i], orientation="horizontal",
                            drawedges=False, fraction=1, shrink=0.5, anchor=(0.5, 1), ticks=[0, vmax])
        cbar.set_label(f"Attention weights (token #{i+1})")
        cbar.ax.set_xticklabels(['0', '{:.1f}+'.format(vmax)])
        cbar.ax.xaxis.set_ticks_position('top')
    plt.subplots_adjust(wspace=0.1, hspace=1, left=0.05, right=0.95, top=0.9, bottom=0.05)

    def update(i):
        progressbar.update(1)
        for j in range(token_win):
            img_weights[j].set_data(attention_weights[i, j])
        already_predicted_text = "".join(preds[:token_win*i]) if i > 0 else ""
        pred_list = [already_predicted_text]
        pred_list = [p for p in pred_list]
        texts["already_pred"].set_text(pred_list[0] if len(pred_list) > 0 else "")
        texts["already_pred2"].set_text(pred_list[1] if len(pred_list) > 1 else "")
        new_pred = "".join(format_display_tokens(p) for p in preds[token_win*(i-1):token_win*i]) if i > 0 else ""
        texts["new_pred"].set_text(new_pred)
        texts["iter"].set_text("Iteration: {}".format(min(i, len(preds) - 2)))
        return img_weights

    if output_path.split(".")[-1] == "mp4":
        anim = animation.FuncAnimation(fig, update, frames=np.arange(0, num_iter), interval=1000, repeat=False)  # , cache_frame_data=False
        anim.save(output_path, writer='ffmpeg', fps=1, dpi=150)
    else:
        anim = animation.FuncAnimation(fig, update, frames=np.arange(0, num_iter), interval=1000, repeat=True)
        anim.save(output_path, writer='imagemagick', fps=1, dpi=150)
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from dar.generic.utils_visualization import add_custom_color_maps, format_display_tokens


def generate_mt_dan_visualization(input_img, attention_weights, preds, output_path, confidence_scores, dec_mode, dec_static_num, dec_dynamic_threshold):
    add_custom_color_maps()
    matplotlib.use("Agg")

    num_iter = attention_weights.shape[0]
    text_size = 8
    anchor_x, anchor_y = (0.05, 0.95)

    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(9, 10)
    plt.axis('off')
    plt.suptitle("Denis Coquenet, \"MT-DAN\"")

    zero_weights = np.zeros((1, attention_weights.shape[1], attention_weights.shape[2]))
    attention_weights = np.concatenate([zero_weights, attention_weights, zero_weights], axis=0)

    
    ax_img = plt.subplot(gs[:8, :4])
    ax_img.get_yaxis().set_visible(False)
    ax_img.get_xaxis().set_visible(False)
    ax_img.title.set_text('Input document')


    ax_cbar = plt.subplot(gs[8:9, :4])
    ax_cbar.get_yaxis().set_visible(False)
    ax_cbar.get_xaxis().set_visible(False)
    ax_cbar.axis('off')

    ax_pred = plt.subplot(gs[:, 5:])
    ax_pred.get_yaxis().set_visible(False)
    ax_pred.get_xaxis().set_visible(False)
    dec_mode_str = f"threshold: {dec_dynamic_threshold}" if dec_mode == 'dynamic' else f"num kept heads: {dec_static_num}" 
    ax_pred.title.set_text(f"Prediction ({dec_mode}, {dec_mode_str})")

    progressbar = tqdm(total=num_iter, mininterval=0, miniters=0)
    texts = {
        "iter": ax_pred.text(0.5, 0.25, "", size=12, ha='center'),
        "already_pred": ax_pred.text(anchor_x, anchor_y, "", size=text_size, va='top'),
        "already_pred2": ax_pred.text(0.51, 0.99, "", size=text_size, va='top'),
        "new_pred": ax_pred.text(0.5, 0.1, "", size=15, ha='center', va='center', weight="bold"),
    }
    img_input = ax_img.imshow(input_img)
    vmax = 0.2
    norm = Normalize(vmin=0, vmax=vmax)


    img_weights = ax_img.imshow(attention_weights[0], cmap="red_alpha", alpha=1, extent=img_input.get_extent(), norm=norm)

    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap="red_alpha_bar"), ax=ax_cbar, orientation="horizontal",
                        drawedges=False, fraction=1, shrink=0.5, anchor=(0.5, 1), ticks=[0, vmax])
    cbar.set_label(f"Attention weights")
    cbar.ax.set_xticklabels(['0', '{:.1f}+'.format(vmax)])
    plt.subplots_adjust(wspace=0.1, hspace=1, left=0.05, right=0.95, top=0.9, bottom=0.05)

    def compute_last_predictions(i, preds, scores):
        previous_preds = ""
        for j in range(i):
            last_pred_ok = True
            for h in range(len(scores[i])):
                if h == 0 or \
                    dec_mode == "dynamic" and last_pred_ok and confidence_scores[j][h] >= dec_dynamic_threshold or \
                    dec_mode == "static" and h < dec_static_num:
                    previous_preds += preds[j][h]
                else:
                    last_pred_ok = False
        return previous_preds

    def update(i):
        progressbar.update(1)
        img_weights.set_data(attention_weights[i])

        if i == 0:
            already_predicted_text = ""
            new_pred_text = ""
            new_pred = ""
        else:
            new_pred_text = [f"Head 1: {format_display_tokens(preds[i-1][0])} ({confidence_scores[i-1][0]:.2f})"]
            new_pred = [format_display_tokens(preds[i-1][0])]
            last_pred_ok = True
            for j in range(1, len(confidence_scores[i-1])):
                new_pred_text.append(f"Head {j+1}: {format_display_tokens(preds[i-1][j])} ({confidence_scores[i-1][j]:.2f})")
                if dec_mode == "dynamic" and confidence_scores[i-1][j] >= dec_dynamic_threshold  and last_pred_ok or \
                    dec_mode == "static" and j < dec_static_num:
                    new_pred.append(format_display_tokens(preds[i-1][j]))
                else:
                    last_pred_ok = False
                
            new_pred = "".join(new_pred)
            new_pred_text = "\n".join(new_pred_text + [new_pred, ])

            already_predicted_text = compute_last_predictions(i-1, preds, confidence_scores)
        pred_list = [already_predicted_text]
        pred_list = [p for p in pred_list]
        texts["already_pred"].set_text(pred_list[0] if len(pred_list) > 0 else "")
        texts["already_pred2"].set_text(pred_list[1] if len(pred_list) > 1 else "")
        texts["new_pred"].set_text(new_pred_text)
        texts["iter"].set_text("Iteration: {}".format(min(i, len(preds) - 2)))
        return img_weights

    if output_path.split(".")[-1] == "mp4":
        anim = animation.FuncAnimation(fig, update, frames=np.arange(0, num_iter+1), interval=1000, repeat=False)  # , cache_frame_data=False
        anim.save(output_path, writer='ffmpeg', fps=1, dpi=150)
    else:
        anim = animation.FuncAnimation(fig, update, frames=np.arange(0, num_iter+1), interval=1000, repeat=True)
        anim.save(output_path, writer='imagemagick', fps=1, dpi=150)
from typing import List, Dict, Tuple, Union, Any

import os
import numpy as np

# save and plot attentions
def save_att_plot(att: np.array, label: List = None, path: str ="att.png") -> None:
    """
    Plot the softmax attention and save the plot.

    Parameters
    ----------
    att: a numpy array with shape [num_decoder_steps, context_length]
    label: a list of labels with length of num_decoder_steps.
    path: the path to save the attention picture
    att = np.array([[0.00565603, 0.994344 ], [0.00560927, 0.9943908 ],
                    [0.00501599, 0.99498403], [0.90557455, 0.1 ]])
    label = ['a', 'b', 'c', 'd']
    save_att_plot(att, label, path='att.png')
    """

    import matplotlib
    matplotlib.use('Agg') # without using x server
    import matplotlib.pyplot as plt

    decoder_length, encoder_length = att.shape # num_decoder_time_steps, context_length

    fig, ax = plt.subplots()
    ax.imshow(att, aspect='auto', origin='lower', cmap='Greys')
    plt.gca().invert_yaxis()
    plt.xlabel("Encoder timestep")
    plt.ylabel("Decoder timestep")
    plt.xticks(range(encoder_length))
    plt.yticks(range(decoder_length))
    if label: ax.set_yticklabels(label)

    plt.tight_layout()
    plt.savefig(path, format='png')
    plt.close()

def save_att(info: Dict, att_dir: str) -> Tuple[str, str]:
    """
    Save the metainfo of attention named by its uttid.
    The info is a dict with key of 'att' and 'uttid'.

    The image and npz_file of attention matrix
    with its uttid will be save at att_dir

    The path of the image and npz file is returned.

    attention is a matrix with shape [decoder_length, encoder_length]
    """
    att, uttid = info['att'], info['uttid']
    att_image_path = os.path.join(att_dir, f"{uttid}_att.png")
    save_att_plot(att, label=None, path=att_image_path)
    att_npz_path = os.path.join(att_dir, f"{uttid}_att.npz")
    np.savez(att_npz_path, key=uttid, feat=att)
    return att_image_path, att_npz_path

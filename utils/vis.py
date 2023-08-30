from operator import pos
import numpy as np 
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
from sklearn import metrics
import seaborn as sns

def draw_curves(pos_scores, neg_scores, pos_label, neg_label, score_label, prob_scale=False):
    """
    Draw three curves: distribution, PR-curve, ROC-curve
    """
    print("Drawing the curves")
    fh, axes = plt.subplots(1, 3, figsize=(15, 5))
    # draw distributions
    draw_score_dist(pos_scores, neg_scores, ax=axes[0], pos_label=pos_label, neg_label=neg_label, score_label=score_label, prob_scale=prob_scale)
    # ROC curve
    draw_ROC(pos_scores, neg_scores, axes[1])
    # PR curve
    draw_PR(pos_scores, neg_scores, axes[2])
    plt.show()

def draw_ROC(pos_scores, neg_scores, ax):
    # get TPR and FPR
    scores = np.concatenate((pos_scores, neg_scores))
    labels = np.zeros_like(scores)
    labels[:pos_scores.size] += 1
    fpr,tpr, _ = metrics.roc_curve(labels, scores)

    # plot
    lw = 2
    ax.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve')
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', label="Random guess")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC-curve')
    ax.legend(loc="lower right")    

def draw_PR(pos_scores, neg_scores, ax):
    # get TPR and FPR
    scores = np.concatenate((pos_scores, neg_scores))
    labels = np.zeros_like(scores)
    labels[:pos_scores.size] += 1
    precision,recall, _ = metrics.precision_recall_curve(labels, scores)

    # plot
    lw = 2
    ax.step(recall, precision, color='darkorange',
             lw=lw, label='PR-Curve')
    ax.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--', label="Random guess")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('PR curve')
    ax.legend(loc="lower left")    


def draw_score_dist(pos_scores, neg_scores, ax = None, 
                    bin=50, range=[], alpha=0.5, pos_color='b', neg_color='gray', frequency=True,
                    pos_label="cifar100", neg_label="SVHN", score_label="score", 
                    prob_scale = False):
    """
    Plot the score distribution between a positive scores and a negative scores
    """
    if ax is None:
        fh, ax= plt.subplots()
    ax.set_title("Score Distribution")

    # auto determine the range
    low = np.amin(np.concatenate((pos_scores, neg_scores)))
    high = np.amax(np.concatenate((pos_scores, neg_scores)))
    diff = high - low
    range = (low - 0.1 * diff, high + 0.1*diff)

    vis_hist(pos_scores, ax, bin=bin, range=range, color=pos_color, alpha=alpha, frequency=frequency, label=pos_label, x_label=score_label, prob_scale=prob_scale)
    vis_hist(neg_scores, ax, bin=bin, range=range, color=neg_color, alpha=alpha, frequency=frequency, label=neg_label, x_label=score_label, prob_scale=prob_scale)


def vis_hist(x_vals, ax,
        bin=100, color='b', alpha=0.5, frequency=True,
        kde_kws={},
        label=None, x_label="score",
        xlim = None, ylim = None
    ):
    """
    Plot the histograms of any kind available
    currently only support the histograms of the cosine similarity
    """
    # histogram
    sns.distplot(x_vals, hist=True, ax=ax, color=color, label=label, bins=bin, kde_kws=kde_kws)
    #ax.hist(scores, bins=bin, range=range, color=color, label=label, alpha=alpha, density=frequency)
    ax.set_xlabel(x_label)
    if frequency:
        ylabel="Frequency"
    else:
        ylabel="Data counts"
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right")

    # set axis limit
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])


if __name__=="__main__":
    pos_scores = np.linspace(20, 35, 500)
    neg_scores = np.linspace(-10, 5, 500)
    pos_label = "synthetic positive class"
    neg_label = "synthetic negative class"
    score_label = "synthetic scores"

    draw_curves(pos_scores, neg_scores, pos_label, neg_label, score_label)
    plt.draw()
import matplotlib as mpl
import seaborn as sns


def init_style(sns_style='whitegrid', font_size_base=16, linewdith_base=1.0, font="Times New Roman"):
    sns.set_style(sns_style)
    colors = sns.color_palette('muted')
    mpl.rcParams["font.family"] = font
    mpl.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams["font.size"] = font_size_base
    mpl.rcParams["grid.linewidth"] = linewdith_base / 2.0
    mpl.rcParams["axes.linewidth"] = linewdith_base
    mpl.rcParams['xtick.major.size'] = 4
    mpl.rcParams['xtick.major.width'] = 1.
    mpl.rcParams['ytick.major.size'] = 4
    mpl.rcParams['ytick.major.width'] = 1.
    return colors

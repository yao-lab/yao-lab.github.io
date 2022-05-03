import os
import typing

import latextable
import numpy
import pandas
from fairlearn.metrics import demographic_parity_difference
from matplotlib import pyplot
from matplotlib import rc
from texttable import Texttable

from lib import os_utils
# matplotlib config
from src.common.data.adult import load_adult
from src.common.data.health import load_health
from src.scripts.lp import get_optimal_front

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 20})
rc('text', usetex=True)
rc('figure', figsize=[10.0, 10.0], dpi=100)
rc('axes', titlesize=24, labelsize=24)
rc('xtick', labelsize=24)
rc('ytick', labelsize=24)

# CONSTANTS
SCATTER_MARKERSIZE = 100
TITLE_PAD = 20
COLOR = ["red", "blue", "green", "cyan", "magenta", "orange", "gray"]
MARKER = ["o", "v", "X", "d", "P", "*", "s"]
FIGURES_FOLDER = "plots"
RESULT_FOLDER = {"adult": "result/eval/adult", "health": "result/eval/health"}
METRIC_TO_READ = ["acc", "dp", "auc_micro", "auc", "dp_soft"]
INV_METRIC_TO_READ = ["acc"]
FORMAT = "pdf"
N = 5
SMALL_PLOT_MARKER_ZOOM = 1.5


def get_dataframe_from_results(results):
    data = []
    columns = []
    count = 0
    # remove checksums
    if results.get("checksums") is not None:
        del results["checksums"]
    read_column_name = False
    for idx, (name, result) in enumerate(results.items()):
        if len(result) > 1:
            print(
                "Warning: More than one result found for a folder. Currently just using the first one")

        if len(result) == 0:
            print(f"skipping {name}")
            continue

        if len(result[0].result.keys()) == 0:
            continue
        elif read_column_name is False:
            for key in sorted(result[0].result.keys()):
                for metric in METRIC_TO_READ:
                    columns.append(f"{key}_{metric}")
                    # don't compute std
                    # columns.append(f"{key}_{metric}_std")

            count += len(columns)
            for key in sorted(result[0].params.keys()):
                columns.append(key)
            # print(result[0].params.keys())
            read_column_name = True
            # print(f"Column name read at {idx}")
        # adding result data here
        arr = []
        for model_result in sorted(result[0].result):
            # model_result is model type : nn, logistic reg etc.
            for metric in METRIC_TO_READ:
                if metric in ["dp", "dp_soft"]:
                    val = numpy.max(
                        list(map(lambda k: numpy.mean(k.result[model_result][metric]), result)))
                else:
                    val = numpy.mean(
                        list(map(lambda k: numpy.mean(k.result[model_result][metric]), result)))

                arr.append(val)

        for i in range(count, len(columns)):
            arr.append(result[0].params[columns[i]])
        # read params and add that here
        data.append(arr)
    # breakpoint()
    return pandas.DataFrame(data, columns=columns)


def get_dataframe_from_invariance_results(results):
    data = []
    columns = []
    count = 0
    if results.get("checksums") is not None:
        del results["checksums"]
    for idx, (name, result) in enumerate(results.items()):
        if len(result) > 1:
            print(
                "Warning: More than one result found for a folder. Currently only averaging them.")

        if len(result) == 0:
            print(f"skipping {name}")
            continue

        if idx == 0:
            for key in sorted(result[0].result.keys()):
                for metric in INV_METRIC_TO_READ:
                    columns.append(f"{key}_{metric}")
                    columns.append(f"{key}_{metric}_std")

            count += len(columns)
            for key in sorted(result[0].params.keys()):
                columns.append(key)
            # print(result[0].params.keys())

        # adding result data here
        arr = []
        # breakpoint()
        for model_result in sorted(result[0].result):
            # model_result is model type : nn, logistic reg etc.
            for metric in INV_METRIC_TO_READ:
                val = numpy.mean(
                    list(map(lambda k: numpy.mean(k.result[model_result][metric]), result)))
                std = numpy.std(
                    list(map(lambda k: numpy.mean(k.result[model_result][metric]), result)))
                arr.append(val)
                arr.append(std)

        for i in range(count, len(columns)):
            arr.append(result[0].params[columns[i]])
        # read params and add that here
        data.append(arr)

    return pandas.DataFrame(data, columns=columns)


def domination(x1, x2):
    # determin if x1 dominate x2
    # breakpoint()
    # want greater acc in 0 dim and lower dp in 1 dim
    if x1[0] > x2[0] and x1[1] < x2[1]:
        return True
    if x1[0] >= x2[0] and x1[1] < x2[1]:
        return True
    if x1[0] > x2[0] and x1[1] <= x2[1]:
        return True
    return False


def get_pareto_front(arr: numpy.ndarray, x_dominates_y: typing.Callable = domination):
    pareto_front = []
    for i in arr:
        for j in arr:
            # print(i,j, x_dominates_y(j, i))
            # if j dominate i, we don't want i
            if x_dominates_y(j, i):
                # print("i is dominated by j")
                break
        else:
            pareto_front.append(i)
    return pareto_front


# plots for main paper
def figure267():
    for data in ["adult", "health"]:
        t = numpy.load(f"result/eval/{data}/fcrl.npy", allow_pickle=True).item()
        df_fcrl = get_dataframe_from_results(t)

        t = numpy.load(f"result/eval/{data}/cvib_supervised.npy", allow_pickle=True).item()
        df_cvib = get_dataframe_from_results(t)

        t = numpy.load(f"result/eval/{data}/lag-fairness.npy", allow_pickle=True).item()
        df_lag = get_dataframe_from_results(t)

        t = numpy.load(f"result/eval/{data}/maxent_arl.npy", allow_pickle=True).item()
        df_maxent = get_dataframe_from_results(t)

        t = numpy.load(f"result/eval/{data}/adv_forgetting.npy", allow_pickle=True).item()
        df_adv_forgetting = get_dataframe_from_results(t)

        if data == "adult":
            t = numpy.load(f"result/eval/{data}/laftr.npy", allow_pickle=True).item()
            df_laftr = get_dataframe_from_results(t)
        else:
            df_laftr = None

        for idx, key in enumerate(
                ["nn_1_layer", "random_forest", "logistic_regression", "svm", "nn_2_layer"]):
            figure = pyplot.figure(figsize=(10, 10))
            ax = figure.add_subplot(1, 1, 1)
            method_array = ["Adversarial Forgetting", "FCRL (Ours)", "CVIB", "MIFR",
                            "MaxEnt-ARL", "LAFTR"]
            df_array = [df_adv_forgetting, df_fcrl, df_cvib, df_lag, df_maxent, df_laftr]
            color_array = [6, 0, 1, 2, 3, 5]
            if data != "adult":
                # remove laftr
                method_array.pop()
                df_array.pop()

            for i, (method, dataframe, color_idx) in enumerate(
                    zip(method_array, df_array, color_array)):
                dataframe.plot(kind="scatter", x=f'{key}_normalized_acc',
                               y=f'{key}_normalized_dp', label=method, c="none",
                               # color=COLOR[color_idx],
                               ax=ax, s=SCATTER_MARKERSIZE, alpha=0.7, marker=MARKER[color_idx],
                               edgecolors=COLOR[color_idx], linewidth=2)

            # MLP results
            if data == "adult":
                ax.scatter([0.84546], [0.1846], label="Unfair MLP", s=SCATTER_MARKERSIZE,
                           color=COLOR[4], marker=MARKER[4])
            if data == "health":
                ax.scatter([0.817346], [0.5779], label="Unfair MLP", s=SCATTER_MARKERSIZE,
                           color=COLOR[4], marker=MARKER[4])

            if data == "adult":
                ax.set_xlim(left=.75, )  # right=0.85)
            if data == "health":
                ax.set_xlim(left=0.66, )  # right=0.83)
            ax.set_ylim(bottom=-0.01)

            ax.set_xlabel(f"Accuracy (mean over {N} runs)")
            ax.set_ylabel("$\Delta_{DP}$" + f" (max over {N} runs)")
            ax.legend(fancybox=True, framealpha=0., loc=2)

            ax.grid(alpha=0.4)
            ax.set_title(
                "Accuracy Vs $\Delta_{DP}$" + f" {'(UCI Adult)' if data == 'adult' else '(Heritage Health)'}",
                pad=TITLE_PAD)
            os_utils.safe_makedirs(os.path.join(FIGURES_FOLDER, "main"))
            pyplot.savefig(
                os.path.join(FIGURES_FOLDER, "main", f"all_methods_{data}_{key}.{FORMAT}"),
                bbox_inches='tight')
            pyplot.close()


def figure3():
    # beta dp/acc plots
    # plot with original data
    for data in ["adult", "health"]:
        t = numpy.load(f"result/eval/{data}/fcrl.npy", allow_pickle=True).item()
        df_fcrl = get_dataframe_from_results(t)

        for idx, key in enumerate(["nn_1_layer"]):
            figure = pyplot.figure(figsize=(10, 10))
            ax = figure.add_subplot(1, 1, 1)
            df_fcrl.plot(kind="scatter", y=f'{key}_normalized_acc', x=f'b', c="none",
                         label="Accuracy", edgecolors=COLOR[0], marker=MARKER[0], ax=ax,
                         s=SCATTER_MARKERSIZE, linewidth=2)
            if data == "adult":
                ax.set_ylim(bottom=.75)
            if data == "health":
                ax.set_ylim(bottom=0.66)

            ax1 = ax.twinx()
            df_fcrl.plot(kind="scatter", y=f'{key}_normalized_dp', x=f'b', c="none",
                         label="$\Delta_{DP}$", edgecolors=COLOR[1], ax=ax1, marker=MARKER[1],
                         s=SCATTER_MARKERSIZE, linewidth=2)
            if data == "adult":
                ax1.set_ylim(bottom=0)
            if data == "health":
                ax1.set_ylim(bottom=0)

            ax.set_ylabel(f"Accuracy (mean over {N} runs)")
            ax1.set_ylabel("$\Delta_{DP}$" + f" (max over {N} runs)")
            ax.set_xlabel("$" + "\\" + "beta$")
            ax.set_xscale("log")
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax1.get_legend_handles_labels()
            ax.get_legend().remove()
            ax1.legend(h1 + h2, l1 + l2, loc=0)

            os_utils.safe_makedirs(os.path.join(FIGURES_FOLDER, "main"))
            pyplot.savefig(
                os.path.join(FIGURES_FOLDER, "main", f"beta_variation_{data}_{key}.{FORMAT}"),
                bbox_inches='tight'
            )
            pyplot.close()


# contrastive vs reconstruction
def figure5a():
    data = "adult"
    t = numpy.load("result/eval/ablations/adult/fcrl.npy", allow_pickle=True).item()
    df_fcrl = get_dataframe_from_results(t)

    t = numpy.load("result/eval/ablations/adult/cvae_cc_supervised.npy", allow_pickle=True).item()
    df_cvae_cc = get_dataframe_from_results(t)

    # modify font size
    fontsize = pyplot.rcParams.get("font.size")
    xlabelsize = pyplot.rcParams.get("xtick.labelsize")
    ylabelsize = pyplot.rcParams.get("ytick.labelsize")
    labelsize = pyplot.rcParams.get("axes.labelsize")

    pyplot.rcParams.update(
        {"font.size": 40, "xtick.labelsize": 40, "ytick.labelsize": 40, "axes.labelsize": 40})

    # breakpoint()
    for idx, key in enumerate(["nn_1_layer"]):
        figure = pyplot.figure(figsize=(10, 10))
        ax = figure.add_subplot(1, 1, 1)

        df_fcrl.plot(kind="scatter", x=f'{key}_normalized_acc', y=f'{key}_normalized_dp',
                     label="Contrastive", edgecolors=COLOR[0], c="none", ax=ax,
                     marker=MARKER[0], linewidth=2,
                     s=SCATTER_MARKERSIZE * SMALL_PLOT_MARKER_ZOOM)
        df_cvae_cc.plot(kind="scatter", x=f'{key}_normalized_acc', y=f'{key}_normalized_dp',
                        label="Reconstruction", edgecolors=COLOR[1], c="none", ax=ax,
                        marker=MARKER[1], linewidth=2,
                        s=SCATTER_MARKERSIZE * SMALL_PLOT_MARKER_ZOOM)
        ax.set_ylim(top=0.2, bottom=0)
        ax.set_xlim(right=0.845)
        # ax.set_xlabel(f"Accuracy (mean over {N} runs)")
        # ax.set_ylabel("$\Delta_{DP}$" + f" (max over {N} runs)")
        ax.set_xlabel(f"Accuracy ")
        ax.set_ylabel("$\Delta_{DP}$")

        ax.grid(alpha=0.4)
        ax.legend(fancybox=True, framealpha=0.5)
        pyplot.tight_layout()
        # ax.set_title(
        #     "Contrastive Vs Reconstruction Bound" + f" {'(UCI Adult)' if data == 'adult' else '(Heritage Health)'}")
        os_utils.safe_makedirs(os.path.join(FIGURES_FOLDER, "main"))
        pyplot.savefig(
            os.path.join(FIGURES_FOLDER, "main",
                         f"contrastive_vs_reconstruction_conditional_{data}_{key}.{FORMAT}"),
            bbox_inches='tight'
        )
        pyplot.close()

    # put values back
    pyplot.rcParams.update(
        {"font.size": fontsize, "xtick.labelsize": xlabelsize, "ytick.labelsize": ylabelsize,
         "axes.labelsize": labelsize})


# conditional vs non-conditional
def figure5b():
    data = "adult"
    t = numpy.load("result/eval/ablations/adult/fcrl.npy", allow_pickle=True).item()
    df_fcrl = get_dataframe_from_results(t)

    t = numpy.load("result/eval/ablations/adult/fcrl_no_conditioning.npy", allow_pickle=True).item()
    df_fcrl_no_conditioning = get_dataframe_from_results(t)

    # modify font size
    fontsize = pyplot.rcParams.get("font.size")
    xlabelsize = pyplot.rcParams.get("xtick.labelsize")
    ylabelsize = pyplot.rcParams.get("ytick.labelsize")
    labelsize = pyplot.rcParams.get("axes.labelsize")

    pyplot.rcParams.update(
        {"font.size": 40, "xtick.labelsize": 40, "ytick.labelsize": 40, "axes.labelsize": 40})

    for idx, key in enumerate(["nn_1_layer"]):
        figure = pyplot.figure(figsize=(10, 10))
        ax = figure.add_subplot(1, 1, 1)
        df_fcrl.plot(kind="scatter", x=f'{key}_normalized_acc', y=f'{key}_normalized_dp',
                     label="$I(\\mathbf{y}{:}\\mathbf{z}|\\mathbf{c})$", edgecolors=COLOR[0],
                     c="none", linewidth=2,
                     marker=MARKER[0],
                     ax=ax, s=SCATTER_MARKERSIZE * SMALL_PLOT_MARKER_ZOOM)
        df_fcrl_no_conditioning.plot(kind="scatter", x=f'{key}_normalized_acc',
                                     y=f'{key}_normalized_dp',
                                     label="$I(\\mathbf{y}{:}\\mathbf{z})$", edgecolors=COLOR[1],
                                     c="none",
                                     ax=ax, linewidth=2,
                                     marker=MARKER[1],
                                     s=SCATTER_MARKERSIZE * SMALL_PLOT_MARKER_ZOOM)
        ax.set_xlabel(f"Accuracy ")
        ax.set_ylabel("$\Delta_{DP}$")
        ax.set_ylim(top=0.2, bottom=0)
        ax.set_xlim(right=0.845)
        ax.grid(alpha=0.4)
        ax.legend(fancybox=True, framealpha=0.5)
        pyplot.tight_layout()

        os_utils.safe_makedirs(os.path.join(FIGURES_FOLDER, "main"))

        pyplot.savefig(
            os.path.join(FIGURES_FOLDER, "main",
                         f"conditional_vs_non_conditional_contrastive_{data}_{key}.{FORMAT}"),
            bbox_inches='tight'
        )
        pyplot.close()
    # put values back
    pyplot.rcParams.update(
        {"font.size": fontsize, "xtick.labelsize": xlabelsize,
         "ytick.labelsize": ylabelsize,
         "axes.labelsize": labelsize})


def finetune():
    t = numpy.load("result/eval/adult/fcrl.npy", allow_pickle=True).item()
    df = get_dataframe_from_results(t)

    finetune_results = numpy.load("result/eval/finetune/adult/fcrl.npy", allow_pickle=True).item()
    finetune_results = get_dataframe_from_results(finetune_results)

    figure = pyplot.figure(figsize=(10, 10))
    ax = figure.add_subplot(1, 1, 1)
    finetune_results.plot(kind="scatter", x="nn_1_layer_normalized_acc",
                       y="nn_1_layer_normalized_dp", ax=ax, edgecolors=COLOR[1], c="none",
                       marker=MARKER[1], linewidth=2,
                       label="Training by Finetuning", s=SCATTER_MARKERSIZE)
    df.plot(kind="scatter", x="nn_1_layer_normalized_acc", y="nn_1_layer_normalized_dp",
            ax=ax, s=SCATTER_MARKERSIZE, edgecolors=COLOR[0], label="Training from Scratch",
            marker=MARKER[0], c="none", linewidth=2)

    # get the starting point
    starting_point = df.loc[df["b"] == 0.005]
    starting_point.plot(kind="scatter", x="nn_1_layer_normalized_acc", ax=ax,
                        color=COLOR[2], y="nn_1_layer_normalized_dp", label="Starting Point",
                        marker="P", s=2 * SCATTER_MARKERSIZE)
    ax.set_xlabel(f"Accuracy (mean over {N} runs)")
    ax.set_ylabel("$\Delta_{DP}$" + f" (max over {N} runs)")
    # ax.set_title("Exploring parity-accuracy trade-off by finetuning")
    ax.grid(alpha=0.4)
    os_utils.safe_makedirs(os.path.join(FIGURES_FOLDER, "main"))
    pyplot.savefig(os.path.join(FIGURES_FOLDER, "main", f"adult_finetune.{FORMAT}"),
                   bbox_inches='tight')
    pyplot.close()


def area_over_curve():
    def compute_ideal_area(Y, C):
        len_c = len(numpy.unique(C))
        len_y = len(numpy.unique(Y))
        p_y_c = numpy.zeros((len_y, len_c))

        for c in range(len_c):
            for y in range(len_y):
                p_y_c[y, c] = numpy.logical_and(Y == y, C == c).mean()
        print(p_y_c)

        # compute desired rate i.e p(y=1|C=c)
        desired_rate = p_y_c[1, :].mean()
        errors = p_y_c[1, :] - desired_rate

        majority_acc = max(numpy.mean(Y == 1), 1 - numpy.mean(Y == 1))
        max_dp = demographic_parity_difference(Y, Y, sensitive_features=C)

        area = (1 - majority_acc) * max_dp
        return area, majority_acc, max_dp

    # Methods
    methods = ["fcrl", "cvib_supervised", "lag-fairness", "maxent_arl", "laftr", "adv_forgetting"]

    # compute AUC table
    area = {}
    for data in ["adult", "health"]:
        # compute idea areas
        if data == "adult":
            adult = load_adult(0.2)
            Y = adult["test"][2]
            C = adult["test"][1]
        elif data == "health":
            health = load_health(0.2)
            Y = health["test"][2]
            C = health["test"][1]

        norm_area, majority_acc, max_dp = compute_ideal_area(Y, C)

        print(f"{data}: area: {norm_area}, majority: {majority_acc}, max_dp = {max_dp}")

        area[data] = {}
        for idx, key in enumerate(
                ["nn_1_layer", "nn_2_layer", "random_forest", "logistic_regression", "svm"]):
            area[data][key] = {}
            for m in methods:
                if data == "health" and m == "laftr":
                    continue
                t = numpy.load(f"result/eval/{data}/{m}.npy", allow_pickle=True).item()
                df = get_dataframe_from_results(t)

                # get pareto front
                pareto = df[[f'{key}_normalized_acc', f'{key}_normalized_dp']].values
                # drop nan
                pareto = pareto[~numpy.isnan(pareto).any(axis=1)]
                pareto = get_pareto_front(pareto)
                pareto = numpy.array(pareto)
                pareto = pareto[pareto[:, 1].argsort()]

                # reject points that have more dp than data
                THRESH = 1.0
                idx = pareto.shape[0]
                while idx > -1:
                    if pareto[idx - 1, 1] > THRESH * max_dp:
                        idx = idx - 1
                    else:
                        break
                pareto = pareto[:idx]
                if idx == -1:
                    area[data][key][m] = 0
                    print(f"No point found below dp_max for {m}, {data}")
                    continue

                # add random acc point, 0 (this works as a reference to create horizontal bars
                # add max_dp, pareto[-1,0] i.e max acc you can get at data's dp
                pareto = numpy.concatenate(
                    [[[majority_acc, 0]], pareto, [[pareto[-1, 0], max_dp]]], axis=0)

                # get area by making rectangle
                area[data][key][m] = numpy.sum(
                    # acc                            * dp_next - dp_cur
                    (pareto[:-1, 0] - pareto[0, 0]) * (pareto[1:, 1] - pareto[0:-1, 1]))

                # normalize
                area[data][key][m] /= norm_area

    # dump to table
    for idx, key in enumerate(
            ["nn_1_layer", "nn_2_layer", "random_forest", "logistic_regression", "svm"]):
        table = Texttable()
        table.set_cols_align(["l", "c", "c"])
        table.header(["Method", "UCI Adult", "Heritage Health"])
        for m in methods:
            if m == "fcrl":
                table.add_row(["FCRL (Ours)", area["adult"][key][m], area["health"][key][m]])
            if m == "lag-fairness":
                table.add_row(["MIFR", area["adult"][key][m], area["health"][key][m]])
            if m == "maxent_arl":
                table.add_row(["MaxEnt-ARL", area["adult"][key][m], area["health"][key][m]])
            if m == "cvib_supervised":
                table.add_row(["CVIB", area["adult"][key][m], area["health"][key][m]])
            if m == "laftr":
                table.add_row(["LAFTR", area["adult"][key][m], "N/A"])
            if m == "adv_forgetting":
                table.add_row(
                    ["Adversarial Forgetting", area["adult"][key][m], area["health"][key][m]])

        os_utils.safe_makedirs(os.path.join(FIGURES_FOLDER, "table"))
        with open(os.path.join(FIGURES_FOLDER, "table", f"{key}.tex"), 'w') as f:
            f.write(latextable.draw_latex(table, caption="Area Over Parity Accuracy Curve",
                                          label=f"AOPAC_{key}"))


def area_over_curve_lp():
    def compute_ideal_area(Y, C):
        len_c = len(numpy.unique(C))
        len_y = len(numpy.unique(Y))
        p_y_c = numpy.zeros((len_y, len_c))

        for c in range(len_c):
            for y in range(len_y):
                p_y_c[y, c] = numpy.logical_and(Y == y, C == c).mean()
        print(p_y_c)

        # compute desired rate i.e p(y=1|C=c)
        desired_rate = p_y_c[1, :].mean()
        errors = p_y_c[1, :] - desired_rate

        majority_acc = max(numpy.mean(Y == 1), 1 - numpy.mean(Y == 1))
        max_dp = demographic_parity_difference(Y, Y, sensitive_features=C)

        solution = get_optimal_front(Y, C)
        # add no error and max_dp to the solution
        solution.append([1, max_dp])

        solution = numpy.array(solution)

        # sort by dp
        solution = solution[solution[:, 1].argsort()]

        area = numpy.sum(
            # acc                            * dp_next - dp_cur
            (solution[:-1, 0] - majority_acc) * (solution[1:, 1] - solution[0:-1, 1]))
        return area, majority_acc, max_dp

    # Methods
    methods = ["fcrl", "cvib_supervised", "lag-fairness", "maxent_arl",
               "laftr", "adv_forgetting"]

    # compute AUC table
    area = {}
    for data in ["adult", "health"]:
        # compute idea areas
        if data == "adult":
            adult = load_adult(0.2)
            Y = adult["test"][2]
            C = adult["test"][1]
        elif data == "health":
            health = load_health(0.2)
            Y = health["test"][2]
            C = health["test"][1]

        norm_area, majority_acc, max_dp = compute_ideal_area(Y, C)

        area[data] = {}
        for idx, key in enumerate(
                ["nn_1_layer", "nn_2_layer", "random_forest", "svm", "logistic_regression"]):
            area[data][key] = {}
            for m in methods:
                if data == "health" and m == "laftr":
                    continue
                t = numpy.load(f"result/eval/{data}/{m}.npy", allow_pickle=True).item()
                df = get_dataframe_from_results(t)

                # get pareto front
                pareto = df[[f'{key}_normalized_acc', f'{key}_normalized_dp']].values
                # drop nan
                pareto = pareto[~numpy.isnan(pareto).any(axis=1)]
                pareto = get_pareto_front(pareto)
                pareto = numpy.array(pareto)
                pareto = pareto[pareto[:, 1].argsort()]

                # reject points that have more dp than data
                THRESH = 1.0
                idx = pareto.shape[0]
                while idx > -1:
                    if pareto[idx - 1, 1] > THRESH * max_dp:
                        idx = idx - 1
                    else:
                        break
                pareto = pareto[:idx]
                if idx == -1:
                    area[data][key][m] = 0
                    print(f"No point found below dp_max for {m}, {data}")
                    continue

                # add random acc point, 0 (this works as a reference to create horizontal bars
                # add max_dp, pareto[-1,0] i.e max acc you can get at data's dp
                pareto = numpy.concatenate(
                    [[[majority_acc, 0]], pareto, [[pareto[-1, 0], max_dp]]], axis=0)

                # get area by making rectangle
                area[data][key][m] = numpy.sum(
                    # acc                            * dp_next - dp_cur
                    (pareto[:-1, 0] - pareto[0, 0]) * (pareto[1:, 1] - pareto[0:-1, 1]))

                # normalize
                area[data][key][m] /= norm_area

    # dump to table
    for idx, key in enumerate(
            ["nn_1_layer", "nn_2_layer", "random_forest", "svm", "logistic_regression"]):
        table = Texttable()
        table.set_cols_align(["l", "c", "c"])
        table.header(["Method", "UCI Adult", "Heritage Health"])
        for m in methods:
            if m == "fcrl":
                table.add_row(
                    ["FCRL (Ours)", area["adult"][key][m], area["health"][key][m]])
            if m == "lag-fairness":
                table.add_row(["MIFR", area["adult"][key][m], area["health"][key][m]])
            if m == "maxent_arl":
                table.add_row(["MaxEnt-ARL", area["adult"][key][m], area["health"][key][m]])
            if m == "cvib_supervised":
                table.add_row(["CVIB", area["adult"][key][m], area["health"][key][m]])
            if m == "laftr":
                table.add_row(["LAFTR", area["adult"][key][m], "N/A"])
            if m == "adv_forgetting":
                table.add_row(
                    ["Adversarial Forgetting", area["adult"][key][m],
                     area["health"][key][m]])

        os_utils.safe_makedirs(os.path.join(FIGURES_FOLDER, "table"))
        with open(os.path.join(FIGURES_FOLDER, "table", f"{key}.better.tex"), 'w') as f:
            f.write(latextable.draw_latex(table, caption="Area Over Parity Accuracy Curve",
                                          label=f"AOPAC_{key}"))


def figure9():
    def compute_ideal_stats(Y, C):
        len_c = len(numpy.unique(C))
        len_y = len(numpy.unique(Y))
        p_y_c = numpy.zeros((len_y, len_c))

        for c in range(len_c):
            for y in range(len_y):
                p_y_c[y, c] = numpy.logical_and(Y == y, C == c).mean()
        print(p_y_c)

        # compute desired rate i.e p(y=1|C=c)
        desired_rate = p_y_c[1, :].mean()
        errors = p_y_c[1, :] - desired_rate

        majority_acc = max(numpy.mean(Y == 1), 1 - numpy.mean(Y == 1))
        max_dp = demographic_parity_difference(Y, Y, sensitive_features=C)

        return 0, majority_acc, max_dp

    # modify font size
    fontsize = pyplot.rcParams.get("font.size")
    xlabelsize = pyplot.rcParams.get("xtick.labelsize")
    ylabelsize = pyplot.rcParams.get("ytick.labelsize")
    labelsize = pyplot.rcParams.get("axes.labelsize")
    titlesize = pyplot.rcParams.get("axes.titlesize")

    pyplot.rcParams.update(
        {"font.size": 12, "xtick.labelsize": 16, "ytick.labelsize": 16, "axes.labelsize": 16,
         "axes.titlesize": 20})

    for data in ["adult", "health"]:
        # compute idea areas
        if data == "adult":
            adult = load_adult(0.2)
            Y = adult["test"][2]
            C = adult["test"][1]
        elif data == "health":
            health = load_health(0.2)
            Y = health["test"][2]
            C = health["test"][1]

        _, RANDOM_ACC, MAX_DP = compute_ideal_stats(Y, C)
        t = numpy.load(f"result/eval/{data}/fcrl.npy", allow_pickle=True).item()
        df = get_dataframe_from_results(t)
        for idx, key in enumerate(["nn_1_layer"]):
            figure = pyplot.figure(figsize=(16, 8))
            ax = figure.add_subplot(1, 1, 1)
            pareto = get_pareto_front(df[[f'{key}_normalized_acc', f'{key}_normalized_dp']].values)
            pareto = numpy.array(pareto)
            pareto = pareto[pareto[:, 1].argsort()]

            # plot the points
            df.plot(kind="scatter", x=f'{key}_normalized_acc', y=f'{key}_normalized_dp',
                    c="none", edgecolors=COLOR[0], linewidth=2, marker=MARKER[0], ax=ax,
                    s=SCATTER_MARKERSIZE, label='All Models')
            ax.scatter(pareto[:, 0], pareto[:, 1], label="Pareto Front", c="none",
                       edgecolors=COLOR[1], linewidth=2, marker=MARKER[1],
                       s=SCATTER_MARKERSIZE)

            # create bars
            ax.barh(y=pareto[:-1, 1], width=pareto[:-1, 0] - RANDOM_ACC,
                    height=pareto[1:, 1] - pareto[:-1, 1], left=RANDOM_ACC, color="yellow",
                    alpha=0.2, align="edge", edgecolor="red")
            ax.barh(y=pareto[-1, 1], height=MAX_DP - pareto[-1, 1],
                    width=pareto[-1, 0] - RANDOM_ACC, left=RANDOM_ACC, color="yellow",
                    alpha=0.2, align="edge", edgecolor="red")

            # ideal plot
            ax.plot([1, 1], [MAX_DP, 0], color="red", label="Ideal")

            # ideal plot but better
            solution = get_optimal_front(Y, C)
            solution.append([1, MAX_DP])
            solution = numpy.array(solution)
            solution = solution[solution[:, 1].argsort()]
            ax.plot(solution[:, 0], solution[:, 1], color="cyan", label="Ideal (LP)")

            # box
            ax.plot([RANDOM_ACC, RANDOM_ACC], [0, MAX_DP], color="gray", linestyle="--")
            ax.plot([RANDOM_ACC, 1], [0, 0], color="gray", linestyle="--")
            ax.plot([RANDOM_ACC, 1], [MAX_DP, MAX_DP], color="gray",
                    linestyle="--")

            ax.set_xlabel("Accuracy")
            ax.set_ylabel("$\Delta_{DP}$")
            # ax.set_title(
            # "Acc Vs $\Delta_{DP}$" + f" ({'UCI Adult' if data == 'adult' else 'Heritage Health'})")
            ax.legend()
            ax.set_xlim(left=RANDOM_ACC - 0.005, right=1.005)
            os_utils.safe_makedirs(os.path.join(FIGURES_FOLDER, "appendix"))
            pyplot.savefig(
                os.path.join(FIGURES_FOLDER, "appendix", f"pareto_{data}_{key}.{FORMAT}"),
                bbox_inches='tight'
            )
            pyplot.close()

    # put values back
    pyplot.rcParams.update({"font.size": fontsize, "xtick.labelsize": xlabelsize,
                            "ytick.labelsize": ylabelsize, "axes.labelsize": labelsize,
                            "axes.titlesize": titlesize})


def figure8():
    # maxent: plot for adult data (bn and without bn)
    data="adult"
    key = "nn_1_layer"
    for method in ["maxent_arl", "maxent_arl_bn"]:
        results = numpy.load(f"result/eval/invariance/adult/{method}.npy",allow_pickle=True)

        df = get_dataframe_from_invariance_results(results.item())
        figure = pyplot.figure(figsize=(10, 10))
        ax = figure.add_subplot(1, 1, 1)
        df.plot(ax=ax, kind="scatter", x="b", y=f"{key}_normalized_acc", c="none",
                label="with preprocessing", s=SCATTER_MARKERSIZE, marker=MARKER[0],
                edgecolors=COLOR[0], linewidth=2)
        df.plot(ax=ax, kind="scatter", x="b", y=f"{key}_acc", c="none",
                label="without preprocessing", s=SCATTER_MARKERSIZE, marker=MARKER[1],
                edgecolors=COLOR[1], linewidth=2)

        ax.set_xlabel("$\\alpha$")
        ax.set_ylabel("Accuracy")
        if method == "maxent_arl":
            ax.set_title("MaxEnt ARL", pad=TITLE_PAD)
        if method == "maxent_arl_bn":
            ax.set_title("MaxEnt ARL (Batch Norm)", pad=TITLE_PAD)

        ax.grid(alpha=0.4)
        ax.set_xscale("log")
        if data == "adult":
            ax.set_ylim(0.66, 0.96)

        os_utils.safe_makedirs(os.path.join(FIGURES_FOLDER, "appendix"))
        pyplot.savefig(os.path.join(FIGURES_FOLDER, "appendix",
                                    f"{data}_{method}_invariance_{key}.{FORMAT}"),
                       bbox_inches='tight')

        pyplot.close()

        # adv forgetting: plot for adult data
    data="adult"
    key = "nn_1_layer"
    for method in ["adv_forgetting"]:
        results = numpy.load(f"result/eval/invariance/adult/{method}.npy",   allow_pickle=True)
        df = get_dataframe_from_invariance_results(results.item())
        for r in [1e-3, 1e-2, 1e-1]:
            for l in [1e-3, 1e-2, 1e-1]:
                figure = pyplot.figure(figsize=(10, 10))
                ax = figure.add_subplot(1, 1, 1)

                df_filtered = df.loc[(df["r"] == r) & (df["l"] == l)]

                df_filtered.plot(ax=ax, kind="scatter", x="d",
                                 y=f"{key}_normalized_acc", c="none",
                                 label="with preprocessing", s=SCATTER_MARKERSIZE,
                                 marker=MARKER[0], edgecolors=COLOR[0], linewidth=2)
                df_filtered.plot(ax=ax, kind="scatter", x="d", y=f"{key}_acc", c="none",
                                 label="without preprocessing", s=SCATTER_MARKERSIZE,
                                 marker=MARKER[1], edgecolors=COLOR[1], linewidth=2)

                ax.set_xlabel("$\\delta$")
                ax.set_ylabel("Accuracy")
                if data == "adult":
                    ax.set_ylim(0.66, 0.86)
                ax.set_title("Adversarial Forgetting", pad=TITLE_PAD)
                ax.grid(alpha=0.4)
                ax.set_xscale("log")
                os_utils.safe_makedirs(os.path.join(FIGURES_FOLDER, "appendix"))
                pyplot.savefig(
                    os.path.join(FIGURES_FOLDER, "appendix",
                                 f"{data}_{method}_invariance_{key}_r_{r}_l_{l}.{FORMAT}"),
                    bbox_inches='tight')
                pyplot.close()


if __name__ == "__main__":
    os_utils.safe_makedirs(FIGURES_FOLDER)

    # after running main experiments run this
    figure267()
    area_over_curve()
    area_over_curve_lp()
    figure3()
    figure9()

    # after running ablation experiments run this
    figure5a()
    figure5b()

    # after running finetuning experiment run this
    finetune()

    # after running main experiments run this


    # after running invariance experiments run this
    figure8()
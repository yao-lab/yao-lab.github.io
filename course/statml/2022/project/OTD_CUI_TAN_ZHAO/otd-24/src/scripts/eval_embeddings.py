"""Take a folder... go to all folders and look for config.json"""
import argparse
import importlib
import logging
import os

import numpy
from box import Box
from checksumdir import dirhash
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from lib.src.os_utils import try_cast, safe_makedirs
from src.common.metrics import demographic_parity_difference, demographic_parity_difference_soft


def run_sklearn_model(config, z_train, y_train, z_test, y_test):
    """Config is dictionary that is expected to have following attr
        .name : <name of the model>
        The attr that have the same name as attr of sklearn model are passed to the model as is.
        __<name>__<attr> are caught and passed to method with name <name>
    """

    def get_options(string):
        options = {}
        for (k, v) in config.items():
            if string in k:
                options[k.split(string)[1]] = v
        return options

    try:
        if config.get("preprocess") == "normalize":
            logger.debug("Normalizing")
            model = StandardScaler()
            model.fit(z_train)
            z_train = model.transform(z_train)
            z_test = model.transform(z_test)

        if config.name == "nn":
            options = get_options("__nn__")
            logger.debug(f"Running nn with config {config}")
            model = MLPClassifier(**options)

        if config.name == "logreg":
            options = get_options("__logreg__")
            logger.debug(f"Running logistic regression with config {config}")
            model = LogisticRegression(**options)

        if config.name == "rf":
            options = get_options("__rf__")
            logger.debug(f"Running random forest with config {config}")
            model = RandomForestClassifier(**options)

        if config.name == "svm":
            options = get_options("__svm__")
            logger.debug(f"Running svm with config {config}")
            if not config.get("bagging"):
                # don't use bagging
                model = SVC(probability=True, **options)
            else:
                # use bagging
                n_estimators = 10
                model = BaggingClassifier(SVC(probability=True, **options),
                                          n_estimators=n_estimators, max_samples=1 / n_estimators,
                                          bootstrap=False)
        model.fit(z_train, y_train)
        score = model.score(z_test, y_test)
        prob = model.predict_proba(z_test)
        logger.debug(f"score: {score}")

        return score, prob

    except Exception as e:
        logger.debug(f"Error occured while running sklearn model {e}")
        return None, None


def evaluate_embeddings(folder, target_evaluations, method, existing_result=None):
    if os.path.exists(os.path.join(folder, "embeddings.npy")):
        embedding_path = os.path.join(folder, "embeddings.npy")
    elif os.path.exists(os.path.join(folder, "embedding.npy")):
        embedding_path = os.path.join(folder, "embedding.npy")
    else:
        logger.error(f"Embeddings not found in {folder}")
        return None

    logger.info(f"Loading embedding from {embedding_path}")
    data = numpy.load(embedding_path, allow_pickle=True).item()

    if method == "laftr":
        z_train = numpy.concatenate([data["train"]["Z"], data["valid"]["Z"]], axis=0)
        y_train = numpy.concatenate([data["train"]["Y"], data["valid"]["Y"]], axis=0).reshape(-1)
        c_train = numpy.concatenate([data["train"]["A"], data["valid"]["A"]], axis=0)

        z_test = data["test"]["Z"]
        y_test = data["test"]["Y"].reshape(-1)
        c_test = data["test"]["A"]

    elif method == "lag-fairness":
        z_train = data["train"]["z"]
        y_train = data["train"]["y"].reshape(-1)
        c_train = data["train"]["u"]
        if len(c_train.shape) == 2 and c_train.shape[1] > 1:
            c_train = numpy.argmax(c_train, axis=1)

        z_test = data["test"]["z"]
        y_test = data["test"]["y"].reshape(-1)
        c_test = data["test"]["u"]
        if len(c_test.shape) == 2 and c_test.shape[1] > 1:
            c_test = numpy.argmax(c_test, axis=1)

    elif method == "adv_forgetting":
        z_train = data["z_wave_train"]
        c_train = data["c_train"]
        y_train = data["y_train"]

        z_test = data["z_wave_test"]
        c_test = data["c_test"]
        y_test = data["y_test"]

    else:
        z_train = data["z_train"]
        c_train = data["c_train"]
        y_train = data["y_train"]

        z_test = data["z_test"]
        c_test = data["c_test"]
        y_test = data["y_test"]

    if existing_result is not None:
        result = existing_result
    else:
        result = {}

    # target evaluation here
    if target_evaluations is not None:
        N = target_evaluations.get("num_runs", 1)
        logger.debug(f"Running {N} runs")

        for config in target_evaluations.model_config:
            if result.get(config.friendly_name) is not None:
                acc = result[config.friendly_name]["acc"]
                auc = result[config.friendly_name]["auc"]
                auc_micro = result[config.friendly_name]["auc_micro"]
                dp = result[config.friendly_name]["dp"]
                dp_soft = result[config.friendly_name]["dp_soft"]
                logger.debug(f"Found {len(acc)} runs")
            else:
                acc, pred, auc, auc_micro, dp, dp_soft = [], [], [], [], [], []

            for i in range(N - len(acc)):
                logger.debug(f"Run: {i}")
                acc_, prob_ = run_sklearn_model(config, z_train, y_train, z_test, y_test)
                if acc_ is None:
                    logger.debug("Accuracy was None")
                    continue
                dp_, _ = demographic_parity_difference(y_test, c_test, prob_)
                dp_soft_, _ = demographic_parity_difference_soft(y_test, c_test, prob_)
                auc_ = roc_auc_score(y_test, prob_[:, 1])
                l = len(y_test)
                onehot = numpy.zeros((l, 2))
                onehot[numpy.arange(l), y_test.astype(numpy.int)] = 1
                auc_micro_ = roc_auc_score(onehot, prob_, average="micro")
                acc.append(acc_)
                dp.append(dp_)
                dp_soft.append(dp_soft_)
                auc.append(auc_)
                auc_micro.append(auc_micro_)
                pred.append((prob_, y_test))

            if len(acc) > 0:
                result[config.friendly_name] = {"acc": acc, "auc": auc, "dp": dp,
                                                "dp_soft": dp_soft,
                                                # "pred": pred,
                                                "auc_micro": auc_micro}
    return result


def get_param_from_string(param_folder, method):
    params = {}

    if method == "lag-fairness":
        for name, p in zip(["mi", "e1", "e2", "e3", "e4", "e5"], param_folder.split("-")):
            params[name] = try_cast(p)
        logger.debug(params)
    elif method == "laftr":
        param_folder = ".".join(param_folder.split(".")[:-1])
        # just read the number after g
        param = param_folder.split("_")[-1]
        params["g"] = try_cast(param)
    else:
        for param in param_folder.split("_"):
            if "=" in param:
                params[param.split("=")[0]] = try_cast(param.split("=")[1])
    return params


if __name__ == "__main__":
    # define logger etc
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--debug", action='store_true')
    parser.add_argument("--force", action='store_true', help="force evaluate everything")
    parser.add_argument("-c", "--config", required=True, help="config to evaluate embeddings")

    parser.add_argument("-f", "--folder_path", required=True)
    parser.add_argument("-r", "--result_folder", required=True)

    parser.add_argument(
        "-m", "--method", required=True,
        help="some methods might need special treatment. This is for that and for naming"
    )

    args = parser.parse_args()

    configFile = args.config
    spec = importlib.util.spec_from_file_location("config", configFile)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target_evaluations = module.target_evaluations

    if args.debug:
        logger.setLevel(logging.DEBUG)

    # load already evaluated things. if force is true... we evaluate everything again
    if os.path.exists(f"{args.result_folder}/{args.method}.npy") and args.force is False:
        logger.debug("Found existing evaluations .. using that")
        evaluations = numpy.load(f"{args.result_folder}/{args.method}.npy",
                                 allow_pickle=True).item()
    else:
        evaluations = Box({})  # result array
        evaluations.checksums = Box({})

    for param_folder in os.listdir(args.folder_path):

        # if evaluations exists and checksum matches... we can use the results
        # else ... remove the results and proceed
        if evaluations.get(param_folder) is not None:
            logger.debug(f"{param_folder} already present.. so checksumming")
            # compute hash to compate
            hash = dirhash(os.path.join(args.folder_path, param_folder))
            if hash == evaluations.checksums.get(param_folder):
                logger.debug(
                    f"{param_folder} already present.. and hash matched so reusing the results")
            else:
                logger.debug(f"{param_folder} already present.. but hash mis-matched")
                evaluations.checksums[param_folder] = hash

                # empty the result
                evaluations[param_folder] = []

        # special methods
        if args.method in ["laftr", "lag-fairness", "ml-arl", ]:

            evaluations[param_folder] = []

            r = evaluate_embeddings(os.path.join(args.folder_path, param_folder),
                                    target_evaluations=target_evaluations, method=args.method)
            if r is not None:
                # we have space for config if we need to read but we dont read now
                evaluation = Box({"result": r, "config": {},
                                  "params": get_param_from_string(param_folder, args.method)})
                evaluations[param_folder].append(evaluation)

        else:
            # there will run_<num> folder here, we will store results for all param_folders in an
            # array

            evaluations[param_folder] = []
            for idx, run_folder in enumerate(
                    os.listdir(os.path.join(args.folder_path, param_folder))):
                # get result
                r = evaluate_embeddings(os.path.join(args.folder_path, param_folder, run_folder),
                                        target_evaluations=target_evaluations, method=args.method)
                if r is not None:
                    # we have space for config if we need to read but we dont read now
                    evaluation = Box({"result": r, "config": {},
                                      "params": get_param_from_string(param_folder, args.method)})
                    evaluations[param_folder].append(evaluation)

    safe_makedirs(args.result_folder)
    numpy.save(f"{args.result_folder}/{args.method}", evaluations)

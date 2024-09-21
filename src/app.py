import os
from argparse import ArgumentParser, HelpFormatter
from run import *
import textwrap

INVALID_FILETYPE_MSG = "Error: Invalid file format. %s must be a .csv file."
INVALID_PATH_MSG = "Error: Invalid file path/name. Path %s does not exist."

class RawFormatter(HelpFormatter):
    def _fill_text(self, text, width, indent):
        return "\n".join([textwrap.fill(line, width) for line in textwrap.indent(textwrap.dedent(text), indent).splitlines()])

def validate_file(file_name):
    if not valid_path(file_name):
        print(INVALID_PATH_MSG%(file_name))
        quit()
    elif not valid_filetype(file_name):
        print(INVALID_FILETYPE_MSG%(file_name))
        quit()
    return

def valid_path(file_name):
    return os.path.exists(file_name)

def valid_filetype(file_name):
    return file_name.endswith('.csv')

def classification(args):
    input_path = args.path[0]
    target = args.classify[0]
    output_path = args.dest[0]

    validate_file(input_path)

    print("dataset loaded!")

    files = ["logistic.joblib", "naive_bayes.joblib", "support_vector.joblib", "decision_tree_c.joblib", "random_forest_c.joblib", "gradient_boosting_c.joblib"]
    zip_file_name = output_path

    classification_testing(input_path, target, files, zip_file_name)
    for file in files:
        file_path = file
        if os.path.exists(file_path):
            os.remove(file_path)
        else:
            pass

    print("Classification models ran successfully!")

def clustering(args):
    # input_path = args.path[0]
    # target = args.clustering[0]
    # output_path = args.dest[0]

    # validate_file(input_path)

    # clus = automationizer.Clustering(input_path, target, output_path)
    print("Clustering models ran successfully!")

def feature_selection(args):
    # input_path = args.path[0]
    # feature_matrix = args.selection[0]
    # target_vector = args.selection[1]
    # output_path = args.dest[0]

    # validate_file(input_path)

    # fs = automationizer.FeatureSelection(input_path, feature_matrix, target_vector, output_path)
    print("Feature Selection models ran successfully!")

def regression(args):
    input_path = args.path[0]
    target = args.regress[0]
    output_path = args.dest[0]

    validate_file(input_path)

    print("dataset loaded!")

    files = ["linear.joblib", "lasso.joblib", "decision_tree.joblib", "random_forest.joblib", "gradient_boosting.joblib"]
    zip_file_name = output_path

    regression_testing(input_path, target, files, zip_file_name)
    for file in files:
        file_path = file
        if os.path.exists(file_path):
            os.remove(file_path)
        else:
            pass

    print("Regression models ran successfully!")

def main():

    program_descripton = f'''
        Team Name : machinenotlearning

        Project Idea :

        Automation of the complete machine learning workflow:
        The main goal is to simplify the machine learning workflow for users with different levels of 
        expertise. We ask the user for certain inputs like the dataset file or folder, nature of the dataset, 
        and the type of ML problem. We then perform data preprocessing and feature selection (if necessary), 
        and train relevant models or build neural networks for the data and present a comparative analysis 
        of the models trained, along with downloadable model weight files in any required format.
        '''

    parser = ArgumentParser(description = program_descripton, formatter_class=RawFormatter)

    parser.add_argument("-p", "--path", type = str, nargs=1, metavar="dataset_path", default=None, help="Loads and reads the dataset")

    parser.add_argument("-c", "--classify", type=str, nargs=1, metavar="classification_target", default=None, help="performs classification task on the dataset")

    parser.add_argument("-k", "--clustering", type=str, nargs=1, metavar="clustering_target", default=None, help="performs clustering task on the dataset")

    parser.add_argument("-r", "--regress", type=str, nargs=1, metavar="regression_target", default=None, help="performs regression task on the dataset")

    parser.add_argument("-s", "--selection", type=str, nargs=2, metavar="feature_selection", default=None, help="performs feature selection task on the dataset")

    parser.add_argument("-d", "--dest", type=str, nargs=1, metavar="destination_path", default=None, help="stores the zipped joblib files in the specified location")

    args = parser.parse_args()

    if args.path != None:
        if args.classify != None:
            classification(args)
        elif args.clustering != None:
            clustering(args)
        elif args.selection != None:
            feature_selection(args)
        elif args.regress != None:
            regression(args)

if __name__ == "__main__":
    main()
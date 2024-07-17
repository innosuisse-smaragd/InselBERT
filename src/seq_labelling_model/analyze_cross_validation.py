import os
import json
import statistics

import os
import json
import statistics


def read_validation_values_and_stats(parent_directories):
    results = {}

    # Loop over all provided parent directories
    for parent_directory in parent_directories:
        f1_values = {}
        f1_list = []
        recall_list = []
        precision_list = []
        accuracy_list = []

        # Loop over all items in the current parent directory
        for subdir in os.listdir(parent_directory):
            subdir_path = os.path.join(parent_directory, subdir)

            # Check if the item is a directory
            if os.path.isdir(subdir_path):
                validation_file_path = os.path.join(subdir_path, 'validation_results.json')

                # Check if the validation.json file exists in the subdirectory
                if os.path.isfile(validation_file_path):
                    with open(validation_file_path, 'r') as f:
                        try:
                            data = json.load(f)
                            # Assuming 'value' is the key you are interested in
                            if 'test_f1' in data:
                                f1_values[subdir] = data['test_f1']
                                f1_list.append(data['test_f1'])
                                recall_list.append(data['test_recall'])
                                precision_list.append(data['test_precision'])
                                accuracy_list.append(data['test_accuracy'])
                            else:
                                print(f"'value' key not found in {validation_file_path}")
                        except json.JSONDecodeError:
                            print(f"Error decoding JSON in file: {validation_file_path}")
                else:
                    print(f"validation.json not found in directory: {subdir_path}")

        # Calculate mean and standard deviation
        if f1_list:
            f1_mean_value = statistics.mean(f1_list)
            f1_std_dev_value = statistics.stdev(f1_list)
        else:
            f1_mean_value = None
            f1_std_dev_value = None
        if recall_list:
            recall_mean_value = statistics.mean(recall_list)
            recall_std_dev_value = statistics.stdev(recall_list)
        else:
            recall_mean_value = None
            recall_std_dev_value = None
        if precision_list:
            precision_mean_value = statistics.mean(precision_list)
            precision_std_dev_value = statistics.stdev(precision_list)
        else:
            precision_mean_value = None
            precision_std_dev_value = None
        if accuracy_list:
            accuracy_mean_value = statistics.mean(accuracy_list)
            accuracy_std_dev_value = statistics.stdev(accuracy_list)
        else:
            accuracy_mean_value = None
            accuracy_std_dev_value = None

        # Store results for the current parent directory
        results[parent_directory] = {
            'values_f1': f1_values,
            'mean_f1': f1_mean_value,
            'std_dev_f1': f1_std_dev_value,
            'mean_recall': recall_mean_value,
            'std_dev_recall': recall_std_dev_value,
            'mean_precision': precision_mean_value,
            'std_dev_precision': precision_std_dev_value,
            'mean_accuracy': accuracy_mean_value,
            'std_dev_accuracy': accuracy_std_dev_value
        }

    return results


# Example usage
# parent_directories = ['path/to/parent/directory1', 'path/to/parent/directory2']
# results = read_validation_values_and_stats(parent_directories)
# for parent_dir, stats in results.items():
#     print(f"Parent Directory: {parent_dir}")
#     print(f"Values: {stats['values']}")
#     print(f"Mean: {stats['mean']}")
#     print(f"Standard Deviation: {stats['std_dev']}")


# Example usage
if __name__ == "__main__":
    base = '../../serialized_models/inselbert_seq_labelling/'
    parent_directories = [base + '20240624-094500_medbert/', base + '20240624-115000_inselbert_all/' ,base + '20240624-140900_inselbert_mammo_03/',base + '20240624-160000_inselbert_mammo_10/']
    results = read_validation_values_and_stats(parent_directories)
    for parent_dir, stats in results.items():
        print(f"Parent Directory: {parent_dir}")
        print(f"Values F1: {stats['values_f1']}")
        print(f"Mean F1: {stats['mean_f1']}")
        print(f"Standard Deviation F1: {stats['std_dev_f1']}")
        print(f"Mean Recall: {stats['mean_recall']}")
        print(f"Standard Deviation Recall: {stats['std_dev_recall']}")
        print(f"Mean Precision: {stats['mean_precision']}")
        print(f"Standard Deviation Precision: {stats['std_dev_precision']}")
        print(f"Mean Accuracy: {stats['mean_accuracy']}")
        print(f"Standard Deviation Accuracy: {stats['std_dev_accuracy']}")


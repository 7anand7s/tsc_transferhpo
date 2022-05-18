Informations regarding accessing the benchmark:

The Benchmark of each dataset is saved with its name in the JSON file.

Reading these data is as simple as "pandas.read_json(file_path)"

The column of each Dataframe consists of ['dataset', 'depth', 'nb_filters', 'batch_size', 'kernel_size', 
'use_residual', 'use_bottleneck', 'acc', 'precision', 'recall', 'budget_ran', 'train_curve', 'val_curve', 
'train_curve 1', 'val_curve 1', 'acc 1', 'precision 1', 'recall 1', 
'train_curve 2', 'val_curve 2', 'acc 2', 'precision 2', 'recall 2', 
'train_curve 3', 'val_curve 3', 'acc 3', 'precision 3', 'recall 3', 
'train_curve 4', 'val_curve 4', 'acc 4', 'precision 4', 'recall 4', 
'train_curve 5', 'val_curve 5', 'acc 5', 'precision 5', 'recall 5']

Respective column names can be used to extract data from selected columns. In most cases, all one needs is Hyperparameters & accuracy.

Hyperparameters = df[['depth', 'nb_filters', 'batch_size', 'kernel_size', 'use_residual', 'use_bottleneck']].values
Accuracy	    = df['acc'].values

Do mind the order of Hyperparameters that you need for your experiments!

A shorter version of this benchmark, which contains only the Hyperparameters & Accuracy is stored in the "Sorted" folder. It is sorted with respect to the accuracy.

"RS_sampled_configs" are the hyperparameters this benchmark contains.
If you would like to contribute, please run "RS_sampled_configs_batch2" and extend this benchmark.



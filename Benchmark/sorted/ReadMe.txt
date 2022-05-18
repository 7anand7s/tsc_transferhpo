SORTED from highest accuracy to the lowest along the bottom. 

Dataframe = pandas.read_json(path_to_file, lines=True)

Hyperparameters = df2['config'].values             Ordered in ['depth', 'nb_filters', 'batch_size', 'kernel_size', 'use_residual', 'use_bottleneck']
Accuracy	    = df['accuracy'].values


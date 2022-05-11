
elif sys.argv[1] == 'compare_datasets_b':
    print("are we inside?")

    let_us_start = time.time()
    nb_prototype = 1
    tot_classes = 0
    tot_x_train = []
    tot_y_train = []
    nb_neighbors = 61
    distance_algorithm = 'dtw'
    dist_fun = utils.constants.DISTANCE_ALGORITHMS[distance_algorithm]
    dist_fun_params = utils.constants.DISTANCE_ALGORITHMS_PARAMS[distance_algorithm]

    for archive_name in ARCHIVE_NAMES:
        # read all the datasets
        datasets_dict = read_all_datasets(root_dir, archive_name)
        for dataset_name in ALL_DATASET_NAMES:
            path = '/home/anand7s/Older_tschpo/bigdata18-master2/reduced_datasets/' + dataset_name + '/'
            if not os.path.exists(path):
                raise ValueError('Reduced dataset not found')
            x_train, y_train, x_test, y_test, nb_classes, clases, max_prototypes, \
            init_clusters = read_data_from_dataset(use_init_clusters=True)

            tot_classes += nb_classes

            temp = np.load(path + 'reduced' + dataset_name + '.npy', allow_pickle=True)
            y_t = [dataset_name for _ in range(len(temp))]
            x_t = [i for i in temp]

            tot_x_train = tot_x_train + x_t
            tot_y_train = tot_y_train + y_t

    classes = np.unique(tot_y_train)
    nb_classes = len(classes)

    columns = [('K_' + str(i)) for i in range(1, nb_neighbors + 1)]
    neighbors = pd.DataFrame(data=np.zeros((nb_classes, nb_neighbors),
                                           dtype=np.str_), columns=columns, index=classes)
    neighbors2 = pd.DataFrame(data=np.zeros((nb_classes, nb_neighbors),
                                            dtype=np.str_), columns=columns, index=classes)
    neighbors3 = pd.DataFrame(data=np.zeros((nb_classes, nb_neighbors),
                                            dtype=np.str_), columns=columns, index=classes)

    # to numpy
    tot_y_train = np.array(tot_y_train)
    tot_x_train = np.array(tot_x_train)

    # this is also a loop over the names of the datasets
    for c in classes:
        print(c)
        # get the x_train without the test instances
        x_train = tot_x_train[np.where(tot_y_train != c)]
        # get the y_train without the test instances
        y_train = tot_y_train[np.where(tot_y_train != c)]
        # get the x_test instances
        x_test = tot_x_train[np.where(tot_y_train == c)]
        # init the distances
        distances = []
        new_dist_m = []
        my_dist = {}
        # loop through each test instances
        for x_test_instance in x_test:
            # get the nearest neighbors
            distance_neighbors = get_neighbors(x_train, x_test_instance,
                                               0, dist_fun, dist_fun_params, return_distances=True)
            # concat the distances
            dist_sorted = distance_neighbors.sort(key=operator.itemgetter(1))
            weights = np.zeros([len(distance_neighbors)])
            num = 1
            for i in range(len(distance_neighbors) - 1, 0, -1):
                mul = num / (i ** 3)
                num = num - mul
                weights[i - 1] = mul
            dist_indexes = [distance_neighbors[i][0] for i in range(len(distance_neighbors))]
            dist_values = [distance_neighbors[i][1] * weights[i] for i in range(len(distance_neighbors))]
            temp = list(zip(dist_indexes, dist_values))
            new_dist_m = new_dist_m + temp
            distances = distances + distance_neighbors

        # saving variables for method3
        no_w_d1 = np.array([y_train[distances[i][0]] for i in range(len(distances))])
        no_w_d2 = np.array([distances[i][1] for i in range(len(distances))])

        #method4
        no_wm4_d1 = np.array([y_train[distances[i][0]] for i in range(len(distances))])
        no_wm4_d2 = np.array([distances[i][1] for i in range(len(distances))])
        for iter, value in enumerate(no_wm4_d2, 1):
            no_wm4_d2[iter] = value*(1/iter)
        dict_m4 = {}
        for i, x in enumerate(no_wm4_d1):
            if x not in dict_m4.keys():
                dict_m4[x] = no_wm4_d2[i]
            elif type(dict_m4[x]) == list:
                dict_m4[x].append(no_w_d2[i])
            else:
                dict_m4[x] = [dict_m4[x] + no_wm4_d2[i]]




        # method1
        # sort list by specifying the second item to be sorted on
        distances.sort(key=operator.itemgetter(1))
        # to numpy array the second item only (the label)
        distances = np.array([y_train[distances[i][0]] for i in range(len(distances))])
        # aggregate the closest datasets
        # this is useful if two datasets are in the k nearest neighbors
        # more than once because they have more than one similar class
        distances = pd.unique(distances)
        # leave only the k nearest ones
        for i in range(1, nb_neighbors + 1):
            # get label of the neighbor
            label = distances[i - 1]
            # put the label
            neighbors.loc[c]['K_' + str(i)] = label

        # method2
        new_dist_m.sort(key=operator.itemgetter(1))
        dist1 = np.array([y_train[new_dist_m[i][0]] for i in range(len(new_dist_m))])
        dist2 = np.array([new_dist_m[i][1] for i in range(len(new_dist_m))])
        for ind, dname in enumerate(dist1):
            if dname not in my_dist.keys():
                my_dist[dname] = dist2[ind]
            else:
                my_dist[dname] = my_dist[dname] + dist2[ind]
        sorted_values = sorted(my_dist.values(), reverse=True)  # Sort the values
        sorted_dict = {}
        for i in sorted_values:
            for k in my_dist.keys():
                if my_dist[k] == i:
                    sorted_dict[k] = my_dist[k]
                    break
        for i, k in enumerate(sorted_dict.keys()):
            neighbors2.loc[c]['K_' + str(i + 1)] = k

        # method3
        dict1 = {}
        for i, x in enumerate(no_w_d1):
            if x not in dict1.keys():
                dict1[x] = no_w_d2[i]
            elif type(dict1[x]) == list:
                dict1[x].append(no_w_d2[i])
            else:
                dict1[x] = [dict1[x] + no_w_d2[i]]
        # init lists
        list1 = [0] * len(dict1)
        list2 = [0] * len(dict1)
        list3 = [np.inf] * len(dict1)
        # find the minimum
        for i, x in enumerate(dict1.keys()):
            dict1[x] = sorted(dict1[x])
            list1[i] = x
            list2[i] = dict1[x][0]
            list3[i] = len(dict1[x])
        # save the minimum
        temp = list(zip(list1, list2))
        temp.sort(key=operator.itemgetter(1))
        list1 = [i[0] for i in temp]
        # loop with other values
        list1_new = list1.copy()
        index = 1
        counting = 0
        while True:
            list1 = list1_new.copy()
            if min(list3) < 2:
                break
            for i, x in enumerate(dict1.keys()):
                dict1[x] = sorted(dict1[x])
                list1_new[i] = x
                list2[i] = (list2[i] * 0.50) + (dict1[x][index] * 0.30) + (dict1[x][index + 1] * 0.20)
                list3[i] = len(dict1[x]) - (index + 1)
            temp2 = list(zip(list1_new, list2))
            temp2.sort(key=operator.itemgetter(1))
            list1_new = [i[0] for i in temp2]
            if list1_new[:2] == list1[:2]:
                if counting > 3:
                    break
            if not list1_new[:2] == list1[:2]:
                counting = 0
            index += 2
            counting += 1
        for i in range(1, nb_neighbors + 1):
            # get label of the neighbor
            # put the label
            neighbors3.loc[c]['K_' + str(i)] = list1_new[i - 1]

    neighbors.to_csv(root_dir + 'similar-datasets_hwaz_m.csv')
    neighbors2.to_csv(root_dir + 'similar-datasets_anand1_m.csv')
    neighbors3.to_csv(root_dir + 'similar-datasets_anand2_m3.csv')
    print(neighbors.to_string())
    print(neighbors2.to_string())
    print(time.time() - let_us_start)

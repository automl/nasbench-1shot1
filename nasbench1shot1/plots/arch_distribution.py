import sys
from nasbench1shot1.core.search_spaces import SearchSpace1, SearchSpace2, SearchSpace3


def main(space='3'):
    search_space = eval('SearchSpace'+space)()
    search_space.sample(with_loose_ends=True)

    cs = search_space.get_configuration_space()

    # Load NASBench
    nasbench = NasbenchWrapper(
        'nasbench1shot1/data/nasbench_data/108_e/nasbench_full.tfrecord'
    )
    search_space.objective_function(nasbench, cs.sample_configuration())

    test_error = []
    valid_error = []

    '''
    for i in range(10000):
        adjacency_matrix, node_list = search_space_3.sample_with_loose_ends()
        adjacency_list = adjacency_matrix.astype(np.int).tolist()
        node_list = [INPUT, *node_list, OUTPUT]
        model_spec = api.ModelSpec(matrix=adjacency_list, ops=node_list)
        nasbench.query(model_spec)
    '''
    for adjacency_matrix, ops, model_spec in search_space.generate_search_space_without_loose_ends():
        # Query NASBench
        data = nasbench.query(model_spec)
        for item in data:
            test_error.append(1 - item['test_accuracy'])
            valid_error.append(1 - item['validation_accuracy'])

    print('Number of architectures', len(test_error) / len(data))

    plt.figure()
    plt.title(
        'Distribution of test error in search space (no. architectures{})'.format(
            int(len(test_error) / len(data))
        )
    )
    plt.hist(test_error, bins=800)
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.xlabel('Test error')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.xlim(0, 0.3)
    plt.savefig('nasbench1shot1/plots/plot_export/test_error_distribution_ss{}.pdf'.format(space),
                dpi=600)
    plt.show()

    plt.figure()
    plt.title('Distribution of validation error in search space (no. architectures {})'.format(
        int(len(valid_error) / len(data))))
    plt.hist(valid_error, bins=800)
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.xlabel('Validation error')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.xlim(0, 0.3)
    plt.savefig('nasbench1shot1/plots/plot_export/valid_error_distribution_ss{}.pdf'.format(space),
                dpi=600)
    plt.show()

    print('min test_error', min(test_error), 'min valid_error', min(valid_error))


if __name__ == '__main__':
    space = str(sys.argv[1])
    main(space)


try:
    from clams import grid_search_svm_hyperparams, plot_svm_boundary, \
        plot_lr_boundary
except Exception as e:
    print(e)
    print('Please try: pip install pyCLAMs==0.3.1 or above')
def bs128(args):
    args['batch_size'] = 128
    args['test_batch_size'] = 64
    args['test_no_frames'] = 5
    args['kNN_val'] = 10
    args['fre_filter'] = 50000
    args['fre_cache_filter'] = 5000
    args['fre_valid'] = 5000
    return args


def basic_fix(args):
    args['port'] = 27006
    args['db_name'] = 'vd_unsup_fx'
    args['col_name'] = 'dyn_clstr'
    return args

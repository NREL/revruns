Traceback (most recent call last):
  File "/home/twillia2/.conda-envs/revruns/lib/python3.7/runpy.py", line 193, in _run_module_as_main
    "__main__", mod_spec)
  File "/home/twillia2/.conda-envs/revruns/lib/python3.7/runpy.py", line 85, in _run_code
    exec(code, run_globals)
  File "/home/twillia2/.conda-envs/revruns/lib/python3.7/site-packages/reV/supply_curve/cli_sc_aggregation.py", line 456, in <module>
    main(obj={})
  File "/home/twillia2/.conda-envs/revruns/lib/python3.7/site-packages/click/core.py", line 829, in __call__
    return self.main(*args, **kwargs)
  File "/home/twillia2/.conda-envs/revruns/lib/python3.7/site-packages/click/core.py", line 782, in main
    rv = self.invoke(ctx)
  File "/home/twillia2/.conda-envs/revruns/lib/python3.7/site-packages/click/core.py", line 1259, in invoke
    return _process_result(sub_ctx.command.invoke(sub_ctx))
  File "/home/twillia2/.conda-envs/revruns/lib/python3.7/site-packages/click/core.py", line 1236, in invoke
    return Command.invoke(self, ctx)
  File "/home/twillia2/.conda-envs/revruns/lib/python3.7/site-packages/click/core.py", line 1066, in invoke
    return ctx.invoke(self.callback, **ctx.params)
  File "/home/twillia2/.conda-envs/revruns/lib/python3.7/site-packages/click/core.py", line 610, in invoke
    return callback(*args, **kwargs)
  File "/home/twillia2/.conda-envs/revruns/lib/python3.7/site-packages/click/decorators.py", line 21, in new_func
    return f(get_current_context(), *args, **kwargs)
  File "/home/twillia2/.conda-envs/revruns/lib/python3.7/site-packages/reV/supply_curve/cli_sc_aggregation.py", line 281, in direct
    raise e
  File "/home/twillia2/.conda-envs/revruns/lib/python3.7/site-packages/reV/supply_curve/cli_sc_aggregation.py", line 276, in direct
    check_excl_layers=check_excl_layers)
  File "/home/twillia2/.conda-envs/revruns/lib/python3.7/site-packages/reV/supply_curve/sc_aggregation.py", line 1234, in summary
    check_excl_layers=check_excl_layers, excl_area=excl_area)
  File "/home/twillia2/.conda-envs/revruns/lib/python3.7/site-packages/reV/supply_curve/sc_aggregation.py", line 605, in __init__
    self._gen_index = Aggregation._parse_gen_index(self._gen_fpath)
  File "/home/twillia2/.conda-envs/revruns/lib/python3.7/site-packages/reV/supply_curve/aggregation.py", line 571, in _parse_gen_index
    with Resource(h5_fpath) as f:
  File "/home/twillia2/.conda-envs/revruns/lib/python3.7/site-packages/rex/resource.py", line 426, in __init__
    self._h5 = h5py.File(self.h5_file, 'r')
  File "/home/twillia2/.conda-envs/revruns/lib/python3.7/site-packages/h5py/_hl/files.py", line 408, in __init__
    swmr=swmr)
  File "/home/twillia2/.conda-envs/revruns/lib/python3.7/site-packages/h5py/_hl/files.py", line 173, in make_fid
    fid = h5f.open(name, flags, fapl=fapl)
  File "h5py/_objects.pyx", line 54, in h5py._objects.with_phil.wrapper
  File "h5py/_objects.pyx", line 55, in h5py._objects.with_phil.wrapper
  File "h5py/h5f.pyx", line 88, in h5py.h5f.open
OSError: Unable to open file (unable to open file: name = 'shared-projects/rev/projects/soco/rev/runs/reference/generation/120hh/150ps/150ps_multi-year.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

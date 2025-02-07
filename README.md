---
tags:
- kernel
- mrope
---

# mrope get position ids

This repo is a small rewrite of the [get_position_ids](test/reference.py) function that simply returns the position ids for multimodal input ids.

The goal of this repo is to provide a simplem, close to 1:1 rewrite of the python implementation as a C++ and small CUDA kernel implementation.

```bash
nix develop -L
```

and then running the following command:

```bash
pytest test/test.py -s

# platform linux -- Python 3.12.8, pytest-8.3.3, pluggy-1.5.0
# rootdir: /root/mrope_get_position_ids
# collected 3 items
#
# test/test.py
# Vision config one_segment - Reference time: 131.56 ms
# Vision config one_segment - Extension time: 6.76 ms
#
# .
# Vision config two_segments - Reference time: 1.72 ms
# Vision config two_segments - Extension time: 0.56 ms
#
# .
# Vision config three_segments - Reference time: 2.02 ms
# Vision config three_segments - Extension time: 0.49 ms
#
# .
```


## Notes

- this example isn't that expensive overally, but the kernel avoids the need to copy the data back and forth between the CPU and GPU and allocates all of the values in the tensor in parallel.
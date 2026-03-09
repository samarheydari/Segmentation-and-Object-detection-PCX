import numpy as np
import scipy.signal


def rolling_window(array, window=(0,), asteps=None, wsteps=None, axes=None, toend=True):
    array = np.asarray(array)
    orig_shape = np.asarray(array.shape)
    window = np.atleast_1d(window).astype(int)

    if axes is not None:
        axes = np.atleast_1d(axes)
        w = np.zeros(array.ndim, dtype=int)
        for axis, size in zip(axes, window):
            w[axis] = size
        window = w

    if window.ndim > 1:
        raise ValueError("`window` must be one-dimensional.")
    if np.any(window < 0):
        raise ValueError("All elements of `window` must be larger then 1.")
    if len(array.shape) < len(window):
        raise ValueError("`window` length must be less or equal `array` dimension.")

    _asteps = np.ones_like(orig_shape)
    if asteps is not None:
        asteps = np.atleast_1d(asteps)
        if asteps.ndim != 1:
            raise ValueError("`asteps` must be either a scalar or one dimensional.")
        if len(asteps) > array.ndim:
            raise ValueError("`asteps` cannot be longer then the `array` dimension.")
        _asteps[-len(asteps) :] = asteps

        if np.any(asteps < 1):
            raise ValueError("All elements of `asteps` must be larger then 1.")
    asteps = _asteps

    _wsteps = np.ones_like(window)
    if wsteps is not None:
        wsteps = np.atleast_1d(wsteps)
        if wsteps.shape != window.shape:
            raise ValueError("`wsteps` must have the same shape as `window`.")
        if np.any(wsteps < 0):
            raise ValueError("All elements of `wsteps` must be larger then 0.")

        _wsteps[:] = wsteps
        _wsteps[window == 0] = 1
    wsteps = _wsteps

    if np.any(orig_shape[-len(window) :] < window * wsteps):
        raise ValueError("`window` * `wsteps` larger then `array` in at least one dimension.")

    new_shape = orig_shape

    _window = window.copy()
    _window[_window == 0] = 1

    new_shape[-len(window) :] += wsteps - _window * wsteps
    new_shape = (new_shape + asteps - 1) // asteps
    new_shape[new_shape < 1] = 1
    shape = new_shape

    strides = np.asarray(array.strides)
    strides *= asteps
    new_strides = array.strides[-len(window) :] * wsteps

    if toend:
        new_shape = np.concatenate((shape, window))
        new_strides = np.concatenate((strides, new_strides))
    else:
        _ = np.zeros_like(shape)
        _[-len(window) :] = window
        _window = _.copy()
        _[-len(window) :] = new_strides
        _new_strides = _

        new_shape = np.zeros(len(shape) * 2, dtype=int)
        new_strides = np.zeros(len(shape) * 2, dtype=int)

        new_shape[::2] = shape
        new_strides[::2] = strides
        new_shape[1::2] = _window
        new_strides[1::2] = _new_strides

    new_strides = new_strides[new_shape != 0]
    new_shape = new_shape[new_shape != 0]

    return np.lib.stride_tricks.as_strided(array, shape=new_shape, strides=new_strides)


def tile_array(array, xsize=512, ysize=512, overlap=0.1, padding=True):
    dtype = array.dtype
    rows = array.shape[0]
    cols = array.shape[1]
    if array.ndim == 3:
        bands = array.shape[2]
    elif array.ndim == 2:
        bands = 1
    xsteps = int(xsize - (xsize * overlap))
    ysteps = int(ysize - (ysize * overlap))

    if padding is True:
        ypad = ysize + 1
        xpad = xsize + 1
        array = np.pad(
            array,
            (
                (int(ysize * overlap), ypad + int(ysize * overlap)),
                (int(xsize * overlap), xpad + int(xsize * overlap)),
                (0, 0),
            ),
            mode="symmetric",
        )
    X_ = rolling_window(array, (xsize, ysize, bands), asteps=(xsteps, ysteps, bands))
    X = []
    for i in range(X_.shape[0]):
        for j in range(X_.shape[1]):
            X.append(X_[i, j, 0, :, :, :])
    return np.asarray(X, dtype=dtype)


def untile_array(array_tiled, target_shape, overlap=0.1, smooth_blending=False):
    dtype = array_tiled.dtype
    rows = target_shape[0]
    cols = target_shape[1]
    bands = target_shape[2]
    xsize = array_tiled.shape[1]
    ysize = array_tiled.shape[2]
    xsteps = int(xsize - (xsize * overlap))
    ysteps = int(ysize - (ysize * overlap))
    array_target = np.zeros(target_shape)
    ypad = ysize + 1
    xpad = xsize + 1
    array_target = np.pad(
        array_target,
        (
            (int(ysize * overlap), ypad + int(ysize * overlap)),
            (int(xsize * overlap), xpad + int(xsize * overlap)),
            (0, 0),
        ),
        mode="symmetric",
    )
    X_ = rolling_window(array_target, (xsize, ysize, bands), asteps=(xsteps, ysteps, bands))
    xtiles = int(X_.shape[0])
    ytiles = int(X_.shape[1])

    if smooth_blending is True:
        if overlap > 0.5:
            raise ValueError("Overlap needs to be <=0.5 when using smooth blending.")
        window1d = scipy.signal.tukey(M=xsize, alpha=overlap * 2)
        window2d = np.expand_dims(np.expand_dims(window1d, axis=1), axis=2)
        window2d = window2d * window2d.transpose(1, 0, 2)
        array_tiled = np.array([tile * window2d for tile in array_tiled])
        t = 0
        xoffset = 0
        for x in range(xtiles):
            yoffset = 0
            for y in range(ytiles):
                array_target[
                    xoffset * xsteps : xoffset * xsteps + xsize, yoffset * ysteps : yoffset * ysteps + ysize, :
                ] = (
                    array_target[
                        xoffset * xsteps : xoffset * xsteps + xsize, yoffset * ysteps : yoffset * ysteps + ysize, :
                    ]
                    + array_tiled[t, :, :, :]
                )
                t += 1
                yoffset += 1
            xoffset += 1
    else:
        t = 0
        xoffset = 0
        for x in range(xtiles):
            yoffset = 0
            for y in range(ytiles):
                array_target[
                    xoffset * xsteps : xoffset * xsteps + xsize, yoffset * ysteps : yoffset * ysteps + ysize, :
                ] = array_tiled[t, :, :, :]
                t += 1
                yoffset += 1
            xoffset += 1
    array_target = array_target[
        int(ysize * overlap) : int(ysize * overlap) + rows, int(xsize * overlap) : int(xsize * overlap) + cols, :
    ]
    return array_target.astype(dtype)

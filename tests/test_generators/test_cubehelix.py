import math

import pytest

from color_systems.generators import cubehelix


ch5_float = ((
    0.0, 0.0, 0.0,
), (
    0.40170897897857355, 0.19010234047407282, 0.15751843750000016,
), (
    0.702278638638098, 0.4201364539654304, 0.3766912500000002,
), (
    0.9017089789785735, 0.6901023404740728, 0.6575184375000002,
), (
    1.0, 1.0, 1.0,
))

ch5_blue_float = ((
    0.0, 0.0, 0.0,
), (
    0.23606781249999997, 0.22259968750000003, 0.434963125,
), (
    0.48142374999999993, 0.46346625, 0.7466174999999999,
), (
    0.7360678125, 0.7225996875, 0.934963125,
), (
    1.0, 1.0, 1.0,
))

ch5_green_rot_float = ((
    0.0, 0.0, 0.0,
), (
    0.15436702794471174, 0.26876235205336785, 0.41018276501335643,
), (
    0.6837023886380977, 0.38360270396543056, 0.6233087500000003,
), (
    0.8456329720552882, 0.7312376479466322, 0.5898172349866435,
), (
    1.0, 1.0, 1.0,
))

ch5_yellow_rot_sat = ((
    0.0, 0.0, 0.0,
), (
    0.20821632812500002, 0.271245859375, 0.24999999999999997,
), (
    0.49071187499999996, 0.48173312500000004, 0.62330875,
), (
    0.875351015625, 0.686262421875, 0.7500000000000001,
), (
    1.0, 1.0, 1.0,
))

ch5_displaced_hex = ('#000000', '#364539', '#88759C', '#C8C09C', '#FFFFFF')


@pytest.mark.parametrize(
    'shape, kwargs, output',
    [
        pytest.param(
            5,
            {},
            ch5_float,
            id='default',
        ),
        pytest.param(
            5,
            {'hue': 2},
            ch5_blue_float,
            id='blue',
        ),
        pytest.param(
            5,
            {'reverse': True},
            ch5_float[::-1],
            id='reverse',
        ),
        pytest.param(
            5,
            {
                'hue': 1,
                'rotations': 1,
            },
            ch5_green_rot_float,
            id='green_rot',
        ),
        pytest.param(
            5,
            {
                'hue': 0.5,
                'rotations': 1,
                'saturation': (0, 1),
            },
            ch5_yellow_rot_sat,
            id='yellow_rot_sat',
        ),
    ],
)
def test_cubehelix(shape, kwargs, output):
    code_result = cubehelix(shape, dtype='float', **kwargs)
    comparison = tuple(zip(tuple(map(tuple, code_result)), output))
    assert all(
        math.isclose(x, y, abs_tol=1e-04)
        for i in comparison for x, y in zip(*i)
    )


@pytest.mark.parametrize(
    'shape, kwargs, output',
    [
        pytest.param(
            5,
            {
                'hue': 1.4,
                'rotations': (-1, 0.6),
                'saturation': (0, 1),
            },
            ch5_displaced_hex,
            id='displaced_hex',
        ),
    ],
)
def test_cubehelix_hex(shape, kwargs, output):
    code_result = cubehelix(shape, dtype='hex', **kwargs)
    comparison = tuple(zip(tuple(code_result), output))
    assert all(x[0] == x[1] for x in comparison)

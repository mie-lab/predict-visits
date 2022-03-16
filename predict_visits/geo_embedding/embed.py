import numpy as np

from predict_visits.geo_embedding.sinusoidal import (
    TheoryGridCellSpatialRelationEncoder,
)


def apply_log_coords(coords):
    """
    Apply log to normed coordinates

    Parameters
    ----------
    coords : 2d np array of size (nr_coords x 2)
        normed coordinates
    """
    temp_sign = np.sign(coords)
    log_coords = temp_sign * np.log(np.absolute(coords) + 1)
    return log_coords


def std_log_embedding(coords, std=None):
    """
    Simple baseline embedding using standardized log coordinates

    Parameters
    ----------
    coords : 2d array
        Normalized geographic coordinates
    std : float, optional
        Given normalization factor, by default None, in this case taking the
        standard deviation of the normed coordinates

    Returns
    -------
    ndarray
        Array of normalized coordinates
    """
    log_coords = apply_log_coords(coords)
    if std is None:
        std = np.std(log_coords)
    return log_coords / std, std


def sinusoidal_embedding(coords, frequency, stats=None):
    """
    Sinus embedding proposed by Mai et al (see sinusoidal.py)

    Parameters
    ----------
    coords : 2d array
        Normalized geographic coordinates
    frequency : int
        How detailed the embedding should be --> leads to more dimensions

    Returns
    -------
    2d array
        Embedded coordinates. The embedding dimension is frequency * 6
    """
    if stats is None:
        assert len(coords) > 1
        coords_wo_zero = np.absolute(coords)[coords != 0]
        lambda_min, lambda_max = np.min(coords_wo_zero), np.max(coords_wo_zero)
    else:
        lambda_min, lambda_max = stats
    # print(lambda_min, lambda_max)
    gridcell = TheoryGridCellSpatialRelationEncoder(
        coord_dim=2,
        frequency_num=frequency,
        max_radius=lambda_max,
        min_radius=lambda_min,
        freq_init="geometric",
        ffn=None,
    )

    embedded_coords = gridcell.forward(np.expand_dims(coords, 0))[0]
    return embedded_coords, (lambda_min, lambda_max)

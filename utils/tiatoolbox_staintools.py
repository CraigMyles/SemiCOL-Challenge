import pathlib
import torch
import torch.nn as nn
import pandas as pd
from torchvision.io import read_image
from sklearn.decomposition import DictionaryLearning
from typing import IO, Dict, Optional, Tuple, Union

from skimage import exposure

class FileNotSupported(Exception):
    """Raise No supported file found error."""

    def __init__(self, message="File format is not supported"):
        self.message = message
        super().__init__(self.message)


class MethodNotSupported(Exception):
    """Raise No supported file found error."""

    def __init__(self, message="Method is not supported"):
        self.message = message
        super().__init__(self.message)


def split_path_name_ext(full_path):
    """Split path of a file to directory path, file name and extensions.
    Args:
        full_path (str or pathlib.Path):
            Path to a file.
    Returns:
        tuple:
            Three parts of the input file path:
            - :py:obj:`pathlib.Path` - Parent directory path
            - :py:obj:`str` - File name
            - :py:obj:`list(str)` - File extensions
    Examples:
        >>> from tiatoolbox.utils.misc import split_path_name_ext
        >>> dir_path, file_name, extensions = split_path_name_ext(full_path)
    """
    input_path = pathlib.Path(full_path)
    return input_path.parent.absolute(), input_path.name, input_path.suffixes

def load_stain_matrix(stain_matrix_input):
    """Load a stain matrix as a numpy array.
    Args:
        stain_matrix_input (ndarray or str, pathlib.Path):
            Either a 2x3 or 3x3 numpy array or a path to a saved .npy /
            .csv file. If using a .csv file, there should be no column
            headers provided
    Returns:
        stain_matrix (:class:`numpy.ndarray`):
            The loaded stain matrix.
    Examples:
        >>> from tiatoolbox import utils
        >>> sm = utils.misc.load_stain_matrix(stain_matrix_input)
    """
    if isinstance(stain_matrix_input, (str, pathlib.Path)):
        _, __, suffixes = split_path_name_ext(stain_matrix_input)
        if suffixes[-1] not in [".csv", ".npy"]:
            raise FileNotSupported(
                "If supplying a path to a stain matrix, use either a \
                npy or a csv file"
            )

        if suffixes[-1] == ".csv":
            return pd.read_csv(stain_matrix_input).to_numpy()

        # only other option left for suffix[-1] is .npy
        return np.load(str(stain_matrix_input))

    if isinstance(stain_matrix_input, np.ndarray):
        return stain_matrix_input

    raise TypeError(
        "Stain_matrix must be either a path to npy/csv file or a numpy array"
    )


def contrast_enhancer(img, low_p=2, high_p=98):
    """Enhancing contrast of the input image using intensity adjustment.
       This method uses both image low and high percentiles.
    Args:
        img (:class:`numpy.ndarray`): input image used to obtain tissue mask.
            Image should be uint8.
        low_p (scalar): low percentile of image values to be saturated to 0.
        high_p (scalar): high percentile of image values to be saturated to 255.
            high_p should always be greater than low_p.
    Returns:
        img (:class:`numpy.ndarray`):
            Image (uint8) with contrast enhanced.
    Raises:
        AssertionError: Internal errors due to invalid img type.
    Examples:
        >>> from tiatoolbox import utils
        >>> img = utils.misc.contrast_enhancer(img, low_p=2, high_p=98)
    """
    # check if image is not uint8
    if not img.dtype == np.uint8:
        raise AssertionError("Image should be uint8.")
    img_out = img.copy()
    p_low, p_high = np.percentile(img_out, (low_p, high_p))
    if p_low >= p_high:
        p_low, p_high = np.min(img_out), np.max(img_out)
    if p_high > p_low:
        img_out = exposure.rescale_intensity(
            img_out, in_range=(p_low, p_high), out_range=(0.0, 255.0)
        )
    return np.uint8(img_out)


def vectors_in_correct_direction(e_vectors):
    """Points the eigen vectors in the right direction.
    Args:
        e_vectors (:class:`numpy.ndarray`):
            Eigen vectors.
    Returns:
        :class:`numpy.ndarray`:
            Pointing in the correct direction.
    """
    if e_vectors[0, 0] < 0:
        e_vectors[:, 0] *= -1
    if e_vectors[0, 1] < 0:
        e_vectors[:, 1] *= -1

    return e_vectors


def h_and_e_in_right_order(v1, v2):
    """Rearrange input vectors for H&E in correct order with H as first output.
    Args:
        v1 (:class:`numpy.ndarray`):
            Input vector for stain extraction.
        v2 (:class:`numpy.ndarray`):
            Input vector for stain extraction.
    Returns:
        :class:`numpy.ndarray`:
            Input vectors in the correct order.
    """
    if v1[0] > v2[0]:
        return np.array([v1, v2])

    return np.array([v2, v1])


def dl_output_for_h_and_e(dictionary):
    """Return correct value for H and E from dictionary learning output.
    Args:
        dictionary (:class:`numpy.ndarray`):
            :class:`sklearn.decomposition.DictionaryLearning` output
    Returns:
        :class:`numpy.ndarray`:
            With correct values for H and E.
    """
    if dictionary[0, 0] < dictionary[1, 0]:
        return dictionary[[1, 0], :]

    return dictionary


class CustomExtractor:
    """Get the user-defined stain matrix.
    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield.
    Examples:
        >>> from tiatoolbox.tools.stainextract import CustomExtractor
        >>> from tiatoolbox.utils.misc import imread
        >>> extractor = CustomExtractor(stain_matrix)
        >>> img = imread('path/to/image')
        >>> stain_matrix = extractor.get_stain_matrix(img)
    """

    def __init__(self, stain_matrix):
        self.stain_matrix = stain_matrix
        if self.stain_matrix.shape not in [(2, 3), (3, 3)]:
            raise ValueError("Stain matrix must have shape (2, 3) or (3, 3).")

    def get_stain_matrix(self, _):
        """Get the user defined stain matrix.
        Returns:
            :class:`numpy.ndarray`:
                User defined stain matrix.
        """
        return self.stain_matrix


class RuifrokExtractor:
    """Reuifrok stain extractor.
    Get the stain matrix as defined in:
    Ruifrok, Arnout C., and Dennis A. Johnston. "Quantification of
    histochemical staining by color deconvolution." Analytical and
    quantitative cytology and histology 23.4 (2001): 291-299.
    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield.
    Examples:
        >>> from tiatoolbox.tools.stainextract import RuifrokExtractor
        >>> from tiatoolbox.utils.misc import imread
        >>> extractor = RuifrokExtractor()
        >>> img = imread('path/to/image')
        >>> stain_matrix = extractor.get_stain_matrix(img)
    """

    def __init__(self):
        self.__stain_matrix = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]])

    def get_stain_matrix(self, _):
        """Get the pre-defined stain matrix.
        Returns:
            :class:`numpy.ndarray`:
                Pre-defined  stain matrix.
        """
        return self.__stain_matrix.copy()


class MacenkoExtractor:
    """Macenko stain extractor.
    Get the stain matrix as defined in:
    Macenko, Marc, et al. "A method for normalizing histology
    slides for quantitative analysis." 2009 IEEE International
    Symposium on Biomedical Imaging: From Nano to Macro. IEEE, 2009.
    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield.
    Args:
        luminosity_threshold (float):
            Threshold used for tissue area selection
        angular_percentile (int):
            Percentile of angular coordinates to be selected
            with respect to the principle, orthogonal eigenvectors.
    Examples:
        >>> from tiatoolbox.tools.stainextract import MacenkoExtractor
        >>> from tiatoolbox.utils.misc import imread
        >>> extractor = MacenkoExtractor()
        >>> img = imread('path/to/image')
        >>> stain_matrix = extractor.get_stain_matrix(img)
    """

    def __init__(self, luminosity_threshold=0.8, angular_percentile=99):
        self.__luminosity_threshold = luminosity_threshold
        self.__angular_percentile = angular_percentile

    def get_stain_matrix(self, img):
        """Stain matrix estimation.
        Args:
            img (:class:`numpy.ndarray`):
                Input image used for stain matrix estimation.
        Returns:
            :class:`numpy.ndarray`:
                Estimated stain matrix.
        """
        img = img.astype("uint8")  # ensure input image is uint8
        luminosity_threshold = self.__luminosity_threshold
        angular_percentile = self.__angular_percentile

        # convert to OD and ignore background
        tissue_mask = get_luminosity_tissue_mask(
            img, threshold=luminosity_threshold
        ).reshape((-1,))
        img_od = rgb2od(img).reshape((-1, 3))
        img_od = img_od[tissue_mask]

        # eigenvectors of covariance in OD space (orthogonal as covariance symmetric)
        _, eigen_vectors = np.linalg.eigh(np.cov(img_od, rowvar=False))

        # the two principle eigenvectors
        eigen_vectors = eigen_vectors[:, [2, 1]]

        # make sure vectors are pointing the right way
        eigen_vectors = vectors_in_correct_direction(e_vectors=eigen_vectors)

        # project on this basis.
        proj = np.dot(img_od, eigen_vectors)

        # angular coordinates with respect to the principle, orthogonal eigenvectors
        phi = np.arctan2(proj[:, 1], proj[:, 0])

        # min and max angles
        min_phi = np.percentile(phi, 100 - angular_percentile)
        max_phi = np.percentile(phi, angular_percentile)

        # the two principle colors
        v1 = np.dot(eigen_vectors, np.array([np.cos(min_phi), np.sin(min_phi)]))
        v2 = np.dot(eigen_vectors, np.array([np.cos(max_phi), np.sin(max_phi)]))

        # order of H&E - H first row
        he = h_and_e_in_right_order(v1, v2)

        return he / np.linalg.norm(he, axis=1)[:, None]


class VahadaneExtractor:
    """Vahadane stain extractor.
    Get the stain matrix as defined in:
    Vahadane, Abhishek, et al. "Structure-preserving color normalization
    and sparse stain separation for histological images."
    IEEE transactions on medical imaging 35.8 (2016): 1962-1971.
    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield.
    Args:
        luminosity_threshold (float):
            Threshold used for tissue area selection.
        regularizer (float):
            Regularizer used in dictionary learning.
    Examples:
        >>> from tiatoolbox.tools.stainextract import VahadaneExtractor
        >>> from tiatoolbox.utils.misc import imread
        >>> extractor = VahadaneExtractor()
        >>> img = imread('path/to/image')
        >>> stain_matrix = extractor.get_stain_matrix(img)
    """

    def __init__(self, luminosity_threshold=0.8, regularizer=0.1):
        self.__luminosity_threshold = luminosity_threshold
        self.__regularizer = regularizer

    def get_stain_matrix(self, img):
        """Stain matrix estimation.
        Args:
            img (:class:`numpy.ndarray`):
                Input image used for stain matrix estimation
        Returns:
            :class:`numpy.ndarray`:
                Estimated stain matrix.
        """
        img = img.astype("uint8")  # ensure input image is uint8
        luminosity_threshold = self.__luminosity_threshold
        regularizer = self.__regularizer
        # convert to OD and ignore background
        tissue_mask = get_luminosity_tissue_mask(
            img, threshold=luminosity_threshold
        ).reshape((-1,))
        img_od = rgb2od(img).reshape((-1, 3))
        img_od = img_od[tissue_mask]

        # do the dictionary learning
        dl = DictionaryLearning(
            n_components=2,
            alpha=regularizer,
            transform_alpha=regularizer,
            fit_algorithm="lars",
            transform_algorithm="lasso_lars",
            positive_dict=True,
            verbose=False,
            max_iter=3,
            transform_max_iter=1000,
        )
        dictionary = dl.fit_transform(X=img_od.T).T

        # order H and E.
        # H on first row.
        dictionary = dl_output_for_h_and_e(dictionary)

        return dictionary / np.linalg.norm(dictionary, axis=1)[:, None]

def get_luminosity_tissue_mask(img, threshold):
    """Get tissue mask based on the luminosity of the input image.
    Args:
        img (:class:`numpy.ndarray`):
            Input image used to obtain tissue mask.
        threshold (float):
            Luminosity threshold used to determine tissue area.
    Returns:
        tissue_mask (:class:`numpy.ndarray`):
            Binary tissue mask.
    Examples:
        >>> from tiatoolbox import utils
        >>> tissue_mask = utils.misc.get_luminosity_tissue_mask(img, threshold=0.8)
    """
    img = img.astype("uint8")  # ensure input image is uint8
    img = contrast_enhancer(img, low_p=2, high_p=98)  # Contrast  enhancement
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_lab = img_lab[:, :, 0] / 255.0  # Convert to range [0,1].
    tissue_mask = l_lab < threshold

    # check it's not empty
    if tissue_mask.sum() == 0:
        raise ValueError("Empty tissue mask computed.")

    return tissue_mask

def get_luminosity_tissue_mask(img, threshold):
    """Get tissue mask based on the luminosity of the input image.
    Args:
        img (:class:`numpy.ndarray`):
            Input image used to obtain tissue mask.
        threshold (float):
            Luminosity threshold used to determine tissue area.
    Returns:
        tissue_mask (:class:`numpy.ndarray`):
            Binary tissue mask.
    Examples:
        >>> from tiatoolbox import utils
        >>> tissue_mask = utils.misc.get_luminosity_tissue_mask(img, threshold=0.8)
    """
    img = img.astype("uint8")  # ensure input image is uint8
    img = contrast_enhancer(img, low_p=2, high_p=98)  # Contrast  enhancement
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_lab = img_lab[:, :, 0] / 255.0  # Convert to range [0,1].
    tissue_mask = l_lab < threshold

    # check it's not empty
    if tissue_mask.sum() == 0:
        raise ValueError("Empty tissue mask computed.")

    return tissue_mask

import cv2
import numpy as np

# from tiatoolbox.tools.stainextract import (
#     CustomExtractor,
#     MacenkoExtractor,
#     RuifrokExtractor,
#     VahadaneExtractor,
# )
# from tiatoolbox.utils.exceptions import MethodNotSupported
# from tiatoolbox.utils.misc import load_stain_matrix
# from tiatoolbox.utils.transforms import od2rgb, rgb2od


class StainNormalizer:
    """Stain normalization base class.
    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield.
    Attributes:
        extractor (CustomExtractor, RuifrokExtractor):
            Method specific stain extractor.
        stain_matrix_target (:class:`numpy.ndarray`):
            Stain matrix of target.
        target_concentrations (:class:`numpy.ndarray`):
            Stain concentration matrix of target.
        maxC_target (:class:`numpy.ndarray`):
            99th percentile of each stain.
        stain_matrix_target_RGB (:class:`numpy.ndarray`):
            Target stain matrix in RGB.
    """

    def __init__(self):
        self.extractor = None
        self.stain_matrix_target = None
        self.target_concentrations = None
        self.maxC_target = None
        self.stain_matrix_target_RGB = None

    @staticmethod
    def get_concentrations(img, stain_matrix):
        """Estimate concentration matrix given an image and stain matrix.
        Args:
            img (:class:`numpy.ndarray`):
                Input image.
            stain_matrix (:class:`numpy.ndarray`):
                Stain matrix for haematoxylin and eosin stains.
        Returns:
            numpy.ndarray:
                Stain concentrations of input image.
        """
        od = rgb2od(img).reshape((-1, 3))
        x, _, _, _ = np.linalg.lstsq(stain_matrix.T, od.T, rcond=-1)
        return x.T

    def fit(self, target):
        """Fit to a target image.
        Args:
            target (:class:`numpy.ndarray` of type :class:`numpy.uint8`):
              Target/reference image.
        """
        self.stain_matrix_target = self.extractor.get_stain_matrix(target)
        self.target_concentrations = self.get_concentrations(
            target, self.stain_matrix_target
        )
        self.maxC_target = np.percentile(
            self.target_concentrations, 99, axis=0
        ).reshape((1, 2))
        # useful to visualize.
        self.stain_matrix_target_RGB = od2rgb(self.stain_matrix_target)

    def transform(self, img):
        """Transform an image.
        Args:
            img (:class:`numpy.ndarray` of type :class:`numpy.uint8`):
                RGB input source image.
        Returns:
            :class:`numpy.ndarray`:
                RGB stain normalized image.
        """
        stain_matrix_source = self.extractor.get_stain_matrix(img)
        source_concentrations = self.get_concentrations(img, stain_matrix_source)
        max_c_source = np.percentile(source_concentrations, 99, axis=0).reshape((1, 2))
        source_concentrations *= self.maxC_target / max_c_source
        trans = 255 * np.exp(
            -1 * np.dot(source_concentrations, self.stain_matrix_target)
        )

        # ensure between 0 and 255
        trans[trans > 255] = 255
        trans[trans < 0] = 0

        return trans.reshape(img.shape).astype(np.uint8)


class CustomNormalizer(StainNormalizer):
    """Stain Normalization using a user-defined stain matrix.
    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield.
    Args:
        stain_matrix (:class:`numpy.ndarray`):
            User-defined stain matrix. Must be either 2x3 or 3x3.
    Examples:
        >>> from tiatoolbox.tools.stainnorm import CustomNormalizer
        >>> norm = CustomNormalizer(stain_matrix)
        >>> norm.fit(target_img)
        >>> norm_img = norm.transform(source_img)
    """

    def __init__(self, stain_matrix):
        super().__init__()

        self.extractor = CustomExtractor(stain_matrix)


class RuifrokNormalizer(StainNormalizer):
    """Ruifrok & Johnston stain normalizer.
    Normalize a patch to the stain appearance of the target image using
    the method of:
    Ruifrok, Arnout C., and Dennis A. Johnston. "Quantification of
    histochemical staining by color deconvolution." Analytical and
    quantitative cytology and histology 23.4 (2001): 291-299.
    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield.
    Examples:
        >>> from tiatoolbox.tools.stainnorm import RuifrokNormalizer
        >>> norm = RuifrokNormalizer()
        >>> norm.fit(target_img)
        >>> norm_img = norm.transform(source_img)
    """

    def __init__(self):
        super().__init__()
        self.extractor = RuifrokExtractor()


class MacenkoNormalizer(StainNormalizer):
    """Macenko stain normalizer.
    Normalize a patch to the stain appearance of the target image using
    the method of:
    Macenko, Marc, et al. "A method for normalizing histology slides for
    quantitative analysis." 2009 IEEE International Symposium on
    Biomedical Imaging: From Nano to Macro. IEEE, 2009.
    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield.
    Examples:
        >>> from tiatoolbox.tools.stainnorm import MacenkoNormalizer
        >>> norm = MacenkoNormalizer()
        >>> norm.fit(target_img)
        >>> norm_img = norm.transform(source_img)
    """

    def __init__(self):
        super().__init__()
        self.extractor = MacenkoExtractor()


class VahadaneNormalizer(StainNormalizer):
    """Vahadane stain normalizer.
    Normalize a patch to the stain appearance of the target image using
    the method of:
    Vahadane, Abhishek, et al. "Structure-preserving color normalization
    and sparse stain separation for histological images." IEEE
    transactions on medical imaging 35.8 (2016): 1962-1971.
    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield.
    Examples:
        >>> from tiatoolbox.tools.stainnorm import VahadaneNormalizer
        >>> norm = VahadaneNormalizer()
        >>> norm.fit(target_img)
        >>> norm_img = norm.transform(source_img)
    """

    def __init__(self):
        super().__init__()
        self.extractor = VahadaneExtractor()


class ReinhardNormalizer:
    """Reinhard colour normalizer.
    Normalize a patch colour to the target image using the method of:
    Reinhard, Erik, et al. "Color transfer between images." IEEE
    Computer graphics and applications 21.5 (2001): 34-41.
    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield.
    Attributes:
        target_means (float):
            Mean of each LAB channel.
        target_stds (float):
            Standard deviation of each LAB channel.
    Examples:
        >>> from tiatoolbox.tools.stainnorm import ReinhardNormalizer
        >>> norm = ReinhardNormalizer()
        >>> norm.fit(target_img)
        >>> norm_img = norm.transform(src_img)
    """

    def __init__(self):
        self.target_means = None
        self.target_stds = None

    def fit(self, target):
        """Fit to a target image.
        Args:
            target (:class:`numpy.ndarray` of type :class:`numpy.uint8`):
                Target image.
        """
        means, stds = self.get_mean_std(target)
        self.target_means = means
        self.target_stds = stds

    def transform(self, img):
        """Transform an image.
        Args:
            img (:class:`numpy.ndarray` of type :class:`numpy.uint8`):
                Input image.
        Returns:
            :class:`numpy.ndarray` of type :class:`numpy.float`:
                Colour normalized RGB image.
        """
        chan1, chan2, chan3 = self.lab_split(img)
        means, stds = self.get_mean_std(img)
        norm1 = (
            (chan1 - means[0]) * (self.target_stds[0] / stds[0])
        ) + self.target_means[0]
        norm2 = (
            (chan2 - means[1]) * (self.target_stds[1] / stds[1])
        ) + self.target_means[1]
        norm3 = (
            (chan3 - means[2]) * (self.target_stds[2] / stds[2])
        ) + self.target_means[2]
        return self.merge_back(norm1, norm2, norm3)

    @staticmethod
    def lab_split(img):
        """Convert from RGB uint8 to LAB and split into channels.
        Args:
            img (:class:`numpy.ndarray` of type :class:`numpy.uint8`):
                Input image.
        Returns:
            tuple:
                - :py:obj:`float`:
                    L channel in LAB colour space.
                - :py:obj:`float`:
                    A channel in LAB colour space.
                - :py:obj:`float`:
                    B channel in LAB colour space.
        """
        img = img.astype("uint8")  # ensure input image is uint8
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        img_float = img.astype(np.float32)
        chan1, chan2, chan3 = cv2.split(img_float)
        chan1 /= 2.55  # should now be in range [0,100]
        chan2 -= 128.0  # should now be in range [-127,127]
        chan3 -= 128.0  # should now be in range [-127,127]
        return chan1, chan2, chan3

    @staticmethod
    def merge_back(chan1, chan2, chan3):
        """Take separate LAB channels and merge back to give RGB uint8.
        Args:
            chan1 (float):
                L channel.
            chan2 (float):
                A channel.
            chan3 (float):
                B channel.
        Returns:
            :class:`numpy.ndarray`:
                Merged image.
        """
        chan1 *= 2.55  # should now be in range [0,255]
        chan2 += 128.0  # should now be in range [0,255]
        chan3 += 128.0  # should now be in range [0,255]
        img = np.clip(cv2.merge((chan1, chan2, chan3)), 0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

    def get_mean_std(self, img):
        """Get mean and standard deviation of each channel.
        Args:
            img (:class:`numpy.ndarray` of type :class:`numpy.uint8`):
                Input image.
        Returns:
            tuple:
                - :py:obj:`float` - Means:
                    Mean values for each RGB channel.
                - :py:obj:`float` - Standard deviations:
                    Standard deviation for each RGB channel.
        """
        img = img.astype("uint8")  # ensure input image is uint8
        chan1, chan2, chan3 = self.lab_split(img)
        m1, sd1 = cv2.meanStdDev(chan1)
        m2, sd2 = cv2.meanStdDev(chan2)
        m3, sd3 = cv2.meanStdDev(chan3)
        means = m1, m2, m3
        stds = sd1, sd2, sd3
        return means, stds


def get_normalizer(method_name, stain_matrix=None):
    """Return a :class:`.StainNormalizer` with corresponding name.
    Args:
        method_name (str):
            Name of stain norm method, must be one of "reinhard",
            "custom", "ruifrok", "macenko" or "vahadane".
        stain_matrix (:class:`numpy.ndarray` or str or pathlib.Path):
            User-defined stain matrix. This must either be a numpy array
            or a path to either a .csv or .npy file. This is only
            utilised if using "custom" method name.
    Returns:
        StainNormalizer:
            An object with base :class:'.StainNormalizer' as base class.
    Examples:
        >>> from tiatoolbox.tools.stainnorm import get_normalizer
        >>> norm = get_normalizer('Reinhard')
        >>> norm.fit(target_img)
        >>> norm_img = norm.transform(source_img)
    """
    if method_name.lower() not in [
        "reinhard",
        "ruifrok",
        "macenko",
        "vahadane",
        "custom",
    ]:
        raise MethodNotSupported

    if stain_matrix is not None and method_name.lower() != "custom":
        raise ValueError(
            '`stain_matrix` is only defined when using `method_name`="custom".'
        )

    if method_name.lower() == "reinhard":
        return ReinhardNormalizer()
    if method_name.lower() == "ruifrok":
        return RuifrokNormalizer()
    if method_name.lower() == "macenko":
        return MacenkoNormalizer()
    if method_name.lower() == "vahadane":
        return VahadaneNormalizer()

    return CustomNormalizer(load_stain_matrix(stain_matrix))

def normalize_rows(A):
    return A / torch.norm(A, dim=1, keepdim=True)

def get_macenko_stain_matrix(I, beta=0.15, alpha=1):
    I = I.reshape((-1, 3))
    # OD = -torch.log((I + 1) / 255)
    OD = -torch.log((I.float() + 1) / 255)
    OD = OD[torch.any(OD > beta, dim=1)]
    
    _, eigvecs_complex = torch.linalg.eig(OD.t() @ OD)
    eigvecs = eigvecs_complex.real

    
    eigvecs = eigvecs[:, eigvecs[0].argsort(descending=True)]
    eigvecs = normalize_rows(eigvecs)
    
    if eigvecs[0, 0] < 0:
        eigvecs[:, 0] = -eigvecs[:, 0]
    if eigvecs[0, 1] < 0:
        eigvecs[:, 1] = -eigvecs[:, 1]
    
    that = torch.tensor([alpha, 1]).to(I.device)
    # stain_matrix = eigvecs @ torch.diag(that)
    diag_3x3 = torch.eye(3)
    diag_3x3[:2, :2] = torch.diag(that)
    stain_matrix = eigvecs @ diag_3x3

    
    return stain_matrix

class MacenkoStainNormalization(nn.Module):
    def __init__(self, target_image, alpha=1, beta=0.15):
        super().__init__()
        self.target_image = target_image
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 0

    def __call__(self, batch_images):
        # print("Applying Macenko stain normalization")
        batch_images = batch_images.to(self.target_image.device)

        target_stain_matrix = get_macenko_stain_matrix(self.target_image, alpha=self.alpha, beta=self.beta)

        normalized_batch = []
        for image in batch_images:
            source_stain_matrix = get_macenko_stain_matrix(image, alpha=self.alpha, beta=self.beta)
            source_OD = -torch.log((image.permute(1, 2, 0) + 1) / 255)
            # source_stain_OD = source_OD @ torch.inverse(source_stain_matrix.t())
            source_stain_OD = source_OD @ torch.inverse(source_stain_matrix.t() + torch.eye(source_stain_matrix.size(0)) * self.epsilon)
            target_OD = source_stain_OD @ target_stain_matrix.t()
            target_RGB = 255 * torch.exp(-target_OD)
            normalized_image = torch.clamp(target_RGB.permute(2, 0, 1), 0, 255)
            normalized_batch.append(normalized_image)

        return torch.stack(normalized_batch)


def rgb2od(img):
    r"""Convert from RGB to optical density (:math:`OD_{RGB}`) space.
    .. math::
        RGB = 255 * exp^{-1*OD_{RGB}}
    Args:
        img (:class:`numpy.ndarray` of type :class:`numpy.uint8`):
            RGB image.
    Returns:
        :class:`numpy.ndarray`:
            Optical density (OD) RGB image.
    Examples:
        >>> from tiatoolbox.utils import transforms, misc
        >>> rgb_img = misc.imread('path/to/image')
        >>> od_img = transforms.rgb2od(rgb_img)
    """
    mask = img == 0
    img[mask] = 1
    return np.maximum(-1 * np.log(img / 255), 1e-6)


def od2rgb(od):
    r"""Convert from optical density (:math:`OD_{RGB}`) to RGB.
    .. math::
        RGB = 255 * exp^{-1*OD_{RGB}}
    Args:
        od (:class:`numpy.ndarray`):
            Optical density (OD) RGB image.
    Returns:
        :class:`numpy.ndarray`:
            RGB Image.
    Examples:
        >>> from tiatoolbox.utils import transforms, misc
        >>> rgb_img = misc.imread('path/to/image')
        >>> od_img = transforms.rgb2od(rgb_img)
        >>> rgb_img = transforms.od2rgb(od_img)
    """
    od = np.maximum(od, 1e-6)
    return (255 * np.exp(-1 * od)).astype(np.uint8)


def bounds2locsize(bounds, origin="upper"):
    """Calculate the size of a tuple of bounds.
    Bounds are expected to be in the `(left, top, right, bottom)` or
    `(start_x, start_y, end_x, end_y)` format.
    Args:
        bounds (tuple(int)):
            A 4-tuple or length 4 array of bounds values in `(left, top,
            right, bottom)` format.
        origin (str):
            Upper (Top-left) or lower (bottom-left) origin.
            Defaults to upper.
    Returns:
        tuple:
            A 2-tuple containing integer 2-tuples for location and size:
            - :py:obj:`tuple` - location tuple
                - :py:obj:`int` - x
                - :py:obj:`int` - y
            - :py:obj:`size` - size tuple
                - :py:obj:`int` - width
                - :py:obj:`int` - height
    Examples:
        >>> from tiatoolbox.utils.transforms import bounds2locsize
        >>> bounds = (0, 0, 10, 10)
        >>> location, size = bounds2locsize(bounds)
        >>> from tiatoolbox.utils.transforms import bounds2locsize
        >>> _, size = bounds2locsize((12, 4, 24, 16))
    """
    left, top, right, bottom = bounds
    origin = origin.lower()
    if origin == "upper":
        return np.array([left, top]), np.array([right - left, bottom - top])
    if origin == "lower":
        return np.array([left, bottom]), np.array([right - left, top - bottom])
    raise ValueError("Invalid origin. Only 'upper' or 'lower' are valid.")


def locsize2bounds(location, size):
    """Convert a location and size to bounds.
    Args:
        location (tuple(int)):
            A 2-tuple or length 2 array of x,y coordinates.
        size (tuple(int)):
            A 2-tuple or length 2 array of width and height.
    Returns:
        tuple:
            A tuple of bounds:
            - :py:obj:`int` - left / start_x
            - :py:obj:`int` - top / start_y
            - :py:obj:`int` - right / end_x
            - :py:obj:`int` - bottom / end_y
    """
    return (
        location[0],
        location[1],
        location[0] + size[0],
        location[1] + size[1],
    )


def bounds2slices(
    bounds: Tuple[int, int, int, int],
    stride: Union[int, Tuple[int, int, Tuple[int, int]]] = 1,
) -> Tuple[slice]:
    """Convert bounds to slices.
    Create a tuple of slices for each start/stop pair in bounds.
    Arguments:
        bounds (tuple(int)):
            Iterable of integer bounds. Must be even in length with the
            first half as starting values and the second half as end
            values, e.g. (start_x, start_y, stop_x, stop_y).
        stride (int):
            Stride to apply when converting to slices.
    Returns:
        tuple of slice:
            Tuple of slices in image read order (y, x, channels).
    Example:
        >>> from tiatoolbox.utils.transforms import bounds2slices
        >>> import numpy as np
        >>> bounds = (5, 5, 10, 10)
        >>> array = np.ones((10, 10, 3))
        >>> slices = bounds2slices(bounds)
        >>> region = array[slices, ...]
    """
    if np.size(stride) not in [1, 2]:
        raise ValueError("Invalid stride shape.")
    if np.size(stride) == 1:
        stride = np.tile(stride, 4)
    elif np.size(stride) == 2:  # pragma: no cover
        stride = np.tile(stride, 2)

    start, stop = np.reshape(bounds, (2, -1)).astype(int)
    slice_array = np.stack([start[::-1], stop[::-1]], axis=1)
    return tuple(slice(*x, s) for x, s in zip(slice_array, stride))


def pad_bounds(
    bounds: Tuple[int, int, int, int],
    padding: Union[int, Tuple[int, int], Tuple[int, int, int, int]],
) -> Tuple[int, int, int, int]:
    """Add padding to bounds.
    Arguments:
        bounds (tuple(int)):
            Iterable of integer bounds. Must be even in length with the
            first half as starting values and the second half as end
            values, e.g. (start_x, start_y, stop_x, stop_y).
        padding (int):
            Padding to add to bounds.
    Examples:
        >>> pad_bounds((0, 0, 0, 0), 1)
    Returns:
        tuple of int:
            Tuple of bounds with padding to the edges.
    """
    if np.size(bounds) % 2 != 0:
        raise ValueError("Bounds must have an even number of elements.")
    ndims = np.size(bounds) // 2

    if np.size(padding) not in [1, 2, np.size(bounds)]:
        raise ValueError("Invalid number of padding elements.")

    if np.size(padding) == 1 or np.size(padding) == np.size(bounds):
        pass
    elif np.size(padding) == ndims:  # pragma: no cover
        padding = np.tile(padding, 2)

    signs = np.repeat([-1, 1], ndims)
    return np.add(bounds, padding * signs)
    
target_image = read_image('./target_image.png')
stain_normalizer = MacenkoStainNormalization(target_image)
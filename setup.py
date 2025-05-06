from setuptools import Command, find_packages, setup

__lib_name__ = "STIMGAT"
__lib_version__ = "1.0.0"
__description__ = "STIMGAT: Transcriptome clustering method based on staining images and spatial information"
__url__ = "https://github.com/Talent-L/STIMGAT"
__author__ = "Zhongqiu Shu"
__author_email__ = "shubushizhizhang@163.com"
__license__ = "MIT"
__keywords__ = ["spatial transcriptomics", "Constrasive learning", "Graph Attention Network"]

setup(
    name = __lib_name__,
    version = __lib_version__,
    description = __description__,
    url = __url__,
    author = __author__,
    author_email = __author_email__,
    license = __license__,
    packages = ['STIMGAT'],
    zip_safe = False,
    include_package_data = True,
)

from setuptools import setup, find_packages

setup(
    name='sparse-vessel-masks',
    version='0.1.0',
    description='This library allows the creation of dense voxel labels from sparse annotations like contours and centerlines. '
                'It also provides functionalites for the evaluation of dense voxel labels on these sparse annotations.',
    author='Hinrich Rahlfs',
    author_email='hinrich.rahlfs@dhzc-charite.de',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'tqdm',
        'nibabel',
        'trimesh',
        'networkx',
        'scipy',
        'scikit-image',
        'scikit-learn',
        'shapely',
        'evalutils'
    ],
    extras_require={
        'dev': [
            'flake8',
        ],
    },
    # entry_points={
    #     'console_scripts': [
    #         'run_evaluation=sparse-vessel-masks.run_evaluation:main',
    #         'create_labels=sparse-vessel-masks.main:create_sparse_label'
    #     ],
    # },
)

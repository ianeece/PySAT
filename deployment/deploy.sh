echo "Python version: $1"
source deactivate
conda install conda-build anaconda-client
conda config --set anaconda_upload yes
conda config --env --add channels conda-forge usgs-astrogeology
conda build --token $CONDA_UPLOAD_TOKEN --python $1 recipe

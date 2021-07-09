echo "Python version: $1"
conda deactivate
conda install conda-build anaconda-client
conda config --set anaconda_upload yes
conda config --env --add channels conda-forge 
conda config --env --add channels usgs-astrogeology
conda build --token $CONDA_UPLOAD_TOKEN --python $1 recipe

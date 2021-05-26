call conda create -y -n DLA_python3 python=3.6 pip
call conda install -y -n DLA_python3 -c conda-forge pyimagej openjdk=8 astropy=4.0.2 pillow=8.0.1 scipy=1.5.2
call conda activate DLA_python3
call pip install pyside6==6.0.2 tqdm==4.60.0 scikit-image==0.17.2 pandas==0.25.3
echo Environment DLA_python3 is set. Please use this environment to run program.
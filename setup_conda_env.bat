SET mypath=%~d0
%mypath%
SET mypath=%~p0
cd %mypath%
call conda env create -f DLA_python3.yml
call conda activate DLA_python3
echo Environment DLA_python3 is set. Please use this environment to run program.
SET mypath=%~d0
%mypath%
SET mypath=%~p0
cd %mypath%

echo Try to remove existing ACT_python3 environment.
call conda deactivate
call conda remove -n ACT_python3 --all -y
echo Creating new environment.
call conda env create -f ACT_python3.yml
call conda activate ACT_python3
echo Environment ACT_python3 is set. Please use this environment to run program.

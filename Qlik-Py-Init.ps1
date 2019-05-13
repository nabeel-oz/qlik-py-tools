Write-Output "Setting up the Python virtual environment..."
python -m venv $PSScriptRoot\qlik-py-env
Write-Output "Copying project files to the new directory..."
xcopy /E /I $PSScriptRoot\generated $PSScriptRoot\qlik-py-env\generated
xcopy /E /I $PSScriptRoot\core $PSScriptRoot\qlik-py-env\core
Write-Output "Activating the virtual environment..."
& $PSScriptRoot\qlik-py-env\Scripts\activate.ps1
Write-Output "Installing required packages..."
python -m pip install --upgrade setuptools pip
pip install grpcio grpcio-tools numpy scipy pandas cython joblib
pip install pystan==2.17
pip install fbprophet
pip install scikit-learn==0.20.3
pip install hdbscan
pip install skater==1.1.2
Write-Output "Creating a new firewall rule for TCP port 50055..."
netsh advfirewall firewall add rule name=Qlik-PyTools dir=in action=allow protocol=TCP localport=50055
Write-Output "All done. Run Qlik-Py-Start.bat to start the SSE Extension Service."
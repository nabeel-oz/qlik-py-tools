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
pip install fbprophet==0.4.post2
pip install scikit-learn==0.20.3
pip install hdbscan==0.8.22
pip install skater==1.1.2
pip install spacy==2.1.4
pip install efficient_apriori==1.0.0
python -m spacy download en
Write-Output "Creating a new firewall rule for TCP port 50055..."
netsh advfirewall firewall add rule name=Qlik-PyTools dir=in action=allow protocol=TCP localport=50055
Write-Output "All done. Run Qlik-Py-Start.bat to start the SSE Extension Service."
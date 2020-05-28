@echo off
echo Setting up the Python virtual environment... & echo.
python -m venv "%~dp0\qlik-py-env"
echo.
echo Copying project files to the new directory... & echo.
xcopy /E /I "%~dp0\generated" "%~dp0\qlik-py-env\generated"
xcopy /E /I "%~dp0\core" "%~dp0\qlik-py-env\core"
xcopy /E /I "%~dp0\models" "%~dp0\qlik-py-env\models"
echo.
echo Activating the virtual environment... & echo.
cd /d "%~dp0\qlik-py-env\Scripts"
call activate
cd ..
echo.
echo Installing required packages... & echo.
python -m pip install --upgrade setuptools pip
pip install wheel==0.34.2
pip install grpcio==1.26.0 grpcio-tools==1.26.0 numpy==1.17.5 scipy==1.4.1 pandas==0.25.3 cython==0.29.14 joblib==0.11 holidays==0.9.11 pyyaml==5.3
pip install pystan==2.17
pip install fbprophet==0.4.post2
pip install scikit-learn==0.23.1
pip install hdbscan==0.8.26
pip install spacy==2.2.4
pip install efficient_apriori==1.0.0
pip install tensorflow==1.14.0
pip install keras==2.2.5
python -m spacy download en
echo.
echo Creating a new firewall rule for TCP port 50055... & echo.
netsh advfirewall firewall add rule name="Qlik PyTools" dir=in action=allow protocol=TCP localport=50055
echo.
echo Setup completed. Please check the log above for errors in red text. & echo.
echo Run Qlik-Py-Start.bat to start this Server Side Extension. & echo.
pause

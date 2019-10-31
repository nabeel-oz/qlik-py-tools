@echo off
echo Setting up the Python virtual environment... & echo.
python -m venv "%~dp0\qlik-py-env"
echo.
echo Copying project files to the new directory... & echo.
xcopy /E /I "%~dp0\generated" "%~dp0\qlik-py-env\generated"
xcopy /E /I "%~dp0\core" "%~dp0\qlik-py-env\core"
echo.
echo Activating the virtual environment... & echo.
cd /d "%~dp0\qlik-py-env\Scripts"
call activate
cd ..
echo.
echo Installing required packages... & echo.
python -m pip install --upgrade setuptools pip
pip install grpcio grpcio-tools numpy scipy pandas cython joblib
pip install pystan==2.17
pip install fbprophet==0.4.post2
pip install scikit-learn==0.21.3
pip install hdbscan==0.8.23
pip install skater==1.1.2
pip install spacy==2.1.4
pip install efficient_apriori==1.0.0
pip install tensorflow==1.14.0
pip install keras==2.2.5
python -m spacy download en
echo.
echo Creating a new firewall rule for TCP port 50055... & echo.
netsh advfirewall firewall add rule name="Qlik PyTools" dir=in action=allow protocol=TCP localport=50055
echo.
echo All done. Run Qlik-Py-Start.bat to start the SSE Extension Service. & echo.
pause

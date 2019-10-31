@echo off
echo Setting up the Python virtual environment... & echo.
python -m venv "%~dp0\qlik-py-env"
echo.
echo Copying project files to the new directory... & echo.
xcopy /E /I "%~dp0\generated" "%~dp0\qlik-py-env\generated"
xcopy /E /I "%~dp0\core" "%~dp0\qlik-py-env\core"
xcopy /E /I "%~dp0\offline" "%~dp0\qlik-py-env\offline"
echo.
echo Activating the virtual environment... & echo.
cd /d "%~dp0\qlik-py-env\Scripts"
call activate
cd ..
echo.
echo Installing required packages... & echo.
pip install --no-index --find-links=offline grpcio grpcio-tools numpy scipy pandas cython joblib==0.11
pip install --no-index --find-links=offline pystan==2.17
pip install --no-index --find-links=offline fbprophet==0.4.post2
pip install --no-index --find-links=offline scikit-learn==0.21.3
pip install --no-index --find-links=offline setuptools wheel hdbscan==0.8.23
pip install --no-index --find-links=offline skater==1.1.2
pip install --no-index --find-links=offline spacy==2.1.4
pip install ./offline/en_core_web_sm-2.1.0.tar.gz
pip install --no-index --find-links=offline efficient_apriori==1.0.0
pip install --no-index --find-links=offline tensorflow==1.14.0
pip install --no-index --find-links=offline keras==2.2.5
echo.
echo Creating a new firewall rule for TCP port 50055... & echo.
netsh advfirewall firewall add rule name="Qlik PyTools" dir=in action=allow protocol=TCP localport=50055
echo.
echo All done. Run Qlik-Py-Start.bat to start the SSE Extension Service. & echo.
pause

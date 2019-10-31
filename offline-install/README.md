# Installing PyTools without Internet access

Follow these steps to install this Server-Side Extension (SSE) on an offline Windows machine.

You will need an alternate Windows machine with Internet access to prepare the files, and a way to transfer these files to the target machine.

## Prepare the installation files
Use a Windows machine with Internet access for these steps.

1. Download the Python 3.6 offline [executable installer](https://www.python.org/ftp/python/3.6.8/python-3.6.8-amd64.exe) and copy it to the target machine.
2. Create a layout for the [MS Visual C++ Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2017).
    - Get the [installation file](https://www.visualstudio.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=15) for Visual C++ Build Tools.
    - Use a terminal to create an offline layout for the required components.
        ```
        vs_buildtools__<version id>.exe --layout --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Tools.ARM64 --add Microsoft.VisualStudio.Component.Windows10SDK.15063.Desktop --includeRecommended
        ```
    - Copy the layout folder to the target machine.
    - Detailed instructions are available in the Visual Studio [documentation](https://docs.microsoft.com/en-us/visualstudio/install/create-an-offline-installation-of-visual-studio?view=vs-2017).
3. Download the Python packages required for this project.
    - The Python version on this machine should match the target machine.
    - Download the required packages from a terminal using pip.
        ```
        pip download setuptools wheel grpcio grpcio-tools numpy scipy pandas cython joblib==0.11 pystan==2.17 fbprophet==0.4.post2 scikit-learn==0.21.3 hdbscan==0.8.23 skater==1.1.2 efficient_apriori==1.0.0 spacy==2.1.4 https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.1.0/en_core_web_sm-2.1.0.tar.gz tensorflow==1.14.0 keras==2.2.5
        ```
    - Copy the package files to the target machine into a folder named `offline`.
4. Download the [latest release](https://github.com/nabeel-qlik/qlik-py-tools/releases) for this SSE and copy it to the target machine.
5. Download `Qlik-Py-Init Offline.bat` from the project's repository on GitHub under `offline-install`. 


## Install on the offline machine
1. Install Python 3.6 using the offline executable. Remember to select the option to add Python to your PATH environment variable.
2. Use a terminal to install MS Visual C++ Build Tools from the layout folder prepared earlier.
    ```
    vs_buildtools__<version id>.exe --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Tools.ARM64 --add Microsoft.VisualStudio.Component.Windows10SDK.15063.Desktop --includeRecommended
    ```
3. Extract the latest release for this SSE to a location of your choice. 
4. Place the `offline` folder with the Python package files into the same location.
5. Copy `Qlik-Py-Init Offline.bat` to the same location, right click and chose 'Run as Administrator'. 
6. Continue with the installation steps in the main [documentation](../README.md#installation).

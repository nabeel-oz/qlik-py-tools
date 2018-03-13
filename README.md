# qlik-py-tools
This repository provides a server side extension for Qlik Sense and QlikView built using Python. The intention is to provide a set of functions for data science that can be used as expressions in Qlik. 

The current implementation includes functions for:

- Time series forecasting : Implemented using Facebook Prophet, a modern and robust library for generating quality forecasts.
- Seasonality and holiday analysis: Also using Facebook Prophet.
- Linear correlations : Implemented using Pandas.

Further information on these can be found under [docs](docs).

For more information on Qlik Server Side Extensions see [qlik-oss](https://github.com/qlik-oss/server-side-extension)

**Disclaimer:** This project has been started by me in a personal capacity and is not supported by Qlik. 

### Pre-requisites

- Qlik Sense Enterprise or Desktop
- Python 3.4 or above

### Installation

1. Install Python from [here](https://www.python.org/downloads/). Remember to select the option to add Python to your PATH environment variable.

2. Download this git repository and extract it to a location of your choice. The machine where you are placing this repository should have access to a local or remote Qlik installation.

3. Double click the Qlik-Py-Init.bat in the repository files and let it do it's thing. You can open this file in a text editor to review the commands that will be executed. 

4. Now whenever you want to start this Python service you can run Qlik-Py-Start.bat. 
If you get an error or no output check your firewall's inbound settings. You may need an inbound rule to open up port 50054. If you need to change the port you can do so in the file core\__main__.py by opening the file with a text editor and changing the value of the _DEFAULT_PORT variable.


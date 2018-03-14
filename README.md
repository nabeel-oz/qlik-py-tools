# Python data science tools for Qlik
Qlik's advanced analytics integration provides a path to making modern algorithms more accessible to the wider business audience. This project is an attempt to show what's possible.

This repository provides a server side extension for Qlik Sense built using Python. The intention is to provide a set of functions for data science that can be used as expressions in Qlik. Sample Qlik Sense apps are also included and explained so that the techniques shown here can be easily replicated.

The current implementation includes:

- Time series forecasting : Implemented using [Facebook Prophet](https://research.fb.com/prophet-forecasting-at-scale/), a modern library for easily generating good quality forecasts.
- Seasonality and holiday analysis: Also using Facebook Prophet.
- Linear correlations : Implemented using Pandas.

Further information on these features is available through the Usage section below.

For more information on Qlik Server Side Extensions see [qlik-oss](https://github.com/qlik-oss/server-side-extension).

**Disclaimer:** This project has been started by me in a personal capacity and is not supported by Qlik. 


## Pre-requisites

- Qlik Sense Enterprise or Qlik Sense Desktop
- Python 3.4 or above


## Installation

1. Install Python from [here](https://www.python.org/downloads/). Remember to select the option to add Python to your PATH environment variable.

2. Download this git repository and extract it to a location of your choice. The machine where you are placing this repository should have access to a local or remote Qlik Sense instance.

3. Double click `Qlik-Py-Init.bat` in the repository files and let it do it's thing. You can open this file in a text editor to review the commands that will be executed. If everything goes smoothly you will see a Python virtual environment being set up and some packages being installed. Once the execution completes, do a quick scan of the log to see everything installed correctly. The libraries imported are: `grpcio`, `numpy`, `scipy`, `pandas`, `fbprophet`.

4. Now whenever you want to start this Python service you can run `Qlik-Py-Start.bat`. You should see the message below once the service is up. If you get an error or no output check your firewall's inbound settings. You may need an inbound rule to open up port `50054`. If you need to change the port you can do so in the file `core\__main__.py` by opening the file with a text editor and changing the value of the `_DEFAULT_PORT` variable.

5. Now you need to [set up an Analytics Connection in Qlik Sense Enterprise](https://help.qlik.com/en-US/sense/February2018/Subsystems/ManagementConsole/Content/create-analytic-connection.htm) or [update the Settings.ini file in Qlik Sense Desktop](https://help.qlik.com/en-US/sense/February2018/Subsystems/Hub/Content/Introduction/configure-analytic-connection-desktop.htm).

6. Finally restart the Qlik Sense engine service for Qlik Sense Enterprise or close and reopen Qlik Sense Desktop.
The Python service must be up before the Qlik Sense engine comes up. If a connection between Python and Qlik is established you should see the capabilities listed in the terminal.

![handshake log](docs/images/Run-02.png)


## Usage

We go into the details of each feature in the sections below.

Sample Qlik Sense apps are provided and each app includes a series of Bookmarks that you can step through to see the respective functions in action.

| Documentation | Sample App |
| --- | --- |
| [Prophet](docs/Prophet.md) | [Sample App - Facebook Prophet](docs/Sample_App_Prophet.qvf) |
| [Correlations](docs/Correlation.md) | Work in progress |


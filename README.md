# Python data science tools for Qlik

## Table of Contents

- [Introduction](#introduction)
- [Note on the approach](#note-on-the-approach)
- [Pre-requisites](#pre-requisites)
- [Installation](#installation)
- [Usage](#usage)


## Introduction
Qlik's advanced analytics integration provides a path to making modern algorithms more accessible to the wider business audience. This project is an attempt to show what's possible.

This repository provides a server side extension (SSE) for Qlik Sense built using Python. The intention is to provide a set of functions for data science that can be used as expressions in Qlik. Sample Qlik Sense apps are also included and explained so that the techniques shown here can be easily replicated.

The current implementation includes:

- Clustering : Implemented using [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html), a high performance algorithm that is great for exploratory data analysis. 
- Time series forecasting : Implemented using [Facebook Prophet](https://research.fb.com/prophet-forecasting-at-scale/), a modern library for easily generating good quality forecasts.
- Seasonality and holiday analysis: Also using Facebook Prophet.
- Linear correlations : Implemented using Pandas.

Further information on these features is available through the Usage section below.

For more information on Qlik Server Side Extensions see [qlik-oss](https://github.com/qlik-oss/server-side-extension).

**Disclaimer:** This project has been started by me in a personal capacity and is not supported by Qlik. 


## Note on the approach
In this project we have defined functions that expose open source algorithms to Qlik using the [gRPC framework](http://www.grpc.io/). Each function allows the user to define input data and parameters to control the underlying algorithm's output. 

While native Python script evaluation is possible in Qlik as demonstrated in the [qlik-oss Python examples](https://github.com/qlik-oss/server-side-extension/blob/master/examples/python/GetStarted.md), I have disabled this functionality in this project.

I prefer this approach for two key reasons:
- Separation of the Python implementation from usage in Qlik: App authors in Qlik just need to be able to use the functions, and understand the algorithms at a high level. Any complexity such as handling missing values or scaling the data is abstracted to simple parameters passed in the Qlik expression.
- Security: This server side extension can not be used to execute arbitrary code from Qlik. Users are restricted to the algorithms exposed through this SSE. Security can be further enhanced by running the SSE on a separate, sandboxed machine, and [securing communication with certificates](https://github.com/qlik-oss/server-side-extension/blob/master/generate_certs_guide/README.md).


## Pre-requisites

- Qlik Sense Enterprise or Qlik Sense Desktop
- Python 3.4 or above
- Microsoft Visual C++ Build Tools


## Installation

1. Get Python from [here](https://www.python.org/downloads/). Remember to select the option to add Python to your PATH environment variable.

2. You'll also need a recent C++ compiler such as [Microsoft Visual C++ Build Tools](http://landinghub.visualstudio.com/visual-cpp-build-tools). If you are having trouble finding the correct installer try [this direct link](https://www.visualstudio.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=15).

3. Download this git repository or get the [latest release](https://github.com/nabeel-qlik/qlik-py-tools/releases) and extract it to a location of your choice. The machine where you are placing this repository should have access to a local or remote Qlik Sense instance.

4. Right click `Qlik-Py-Init.bat` and chose 'Run as Administrator'. You can open this file in a text editor to review the commands that will be executed. If everything goes smoothly you will see a Python virtual environment being set up, some packages being installed and TCP Port `50055` being opened for inbound communication. 
     - If you need to change the port you can do so in the file `core\__main__.py` by opening the file with a text editor, changing the value of the `_DEFAULT_PORT` variable, and then saving the file. You will also need to update `Qlik-Py-Init.bat` to use the same port in the `netsh` command. This command will only work if you run the batch file through an elevated command prompt (i.e. with administrator privileges).
     - Once the execution completes, do a quick scan of the log to see everything installed correctly. The libraries imported are: `grpcio`, `numpy`, `scipy`, `pandas`, `fbprophet`, `scikit-learn`, `hdbscan`. Also, check that the `core` and `generated` directories have been moved successfully to the newly created `qlik-py-env` directory.

5. Now whenever you want to start this Python service you can run `Qlik-Py-Start.bat`.

6. Now you need to [set up an Analytics Connection in Qlik Sense Enterprise](https://help.qlik.com/en-US/sense/February2018/Subsystems/ManagementConsole/Content/create-analytic-connection.htm) or [update the Settings.ini file in Qlik Sense Desktop](https://help.qlik.com/en-US/sense/February2018/Subsystems/Hub/Content/Introduction/configure-analytic-connection-desktop.htm).

7. Finally restart the Qlik Sense engine service for Qlik Sense Enterprise or close and reopen Qlik Sense Desktop. This step may not be required if you are using Qlik Sense April 2018. If a connection between Python and Qlik is established you should see the capabilities listed in the terminal.

![handshake log](docs/images/Run-02.png)
*Capabilities may change as this is an ongoing project.*


## Usage

We go into the details of each feature in the sections below.

Sample Qlik Sense apps are provided and each app includes a series of Bookmarks that you can step through to see the respective functions in action.

| Documentation | Sample App | App Dependencies |
| --- | --- | --- |
| [Clustering](docs/Clustering.md) | [Sample App - Clustering with HDBSCAN](docs/Sample_App_Clustering.qvf) | The [qsVariable](https://github.com/erikwett/qsVariable) extension. <br/><br/>Qlik Sense April 2018 or later to view the multi-layered maps. |
| [Prophet](docs/Prophet.md) | [Sample App - Facebook Prophet](docs/Sample_App_Prophet.qvf) | The [qsVariable](https://github.com/erikwett/qsVariable) extension. <br/><br/>Use the bookmarks to step through the sheets with relevant selections. |
| [Correlations](docs/Correlation.md) | [Sample App - Correlations](docs/Sample_App_Correlations.qvf) | None. |

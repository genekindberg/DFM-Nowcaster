# DFM-Nowcaster
A dynamic factor model to nowcast quarterly GDP using many high-frequency series. Implemented in Python

This  is an early implementation of a dynamic factor model nowcasting system, with a similar structure to that used to nowcast global GDP in https://www.bankofengland.co.uk/quarterly-bulletin/2018/2018-q3/gauging-the-globe-the-banks-approach-to-nowcasting-world-gdp

ImportDataPackage.py provides a simple example of using the package "DFM.py" on U.S. GDP data. As long as both files are in the sample folder and the python dependencies are installed, this should work.

Most dependencies are included in Anaconda distributions of Python 3.

The example dataset is very simple, and only estimates 3 monthly data factors and a GDP factor. The monthly data factors only draw on 6 monthly series, but the model has been tested on data containing >130 monthly series and a GDP series.

Any suggestions or mistakes, please contact genekindberg @ googlemail .com

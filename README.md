# qsi-tk

 Data science toolkit (TK) from Quality-Safety research Institute (QSI)

# Installation

> pip install qsi-tk

# Contents

This package is a master library containing various previous packages published by our team.

<table>
    <tbody>
        <tr>
            <td>module</td>
            <td>sub-module</td>
            <td>description</td>
            <td>standalone pypi package</td>
            <td>publication</td>
        </tr>
        <tr>
            <td colspan = 1 rowspan = 3>qsi.io</td>
            <td>
            <td>File I/O, Dataset loading</td>
            <td></td>
            <td>TODO qsi-tk open datasets with algorithms</td>
        </tr>
        <tr>
            <td colspan = 1>qsi.io.aug</td>
            <td>Data augmentation, e.g., generative models</td>
            <td></td>
            <td>TODO Data aug with deep generative models. e.g., " variational autoencoders, generative adversarial networks, autoregressive models, KDE, normalizing flow models, energy-based models, and score-based models. "</td>
        </tr>
        <tr>
            <td>qsi.io.pre</td>
            <td>Data processing, e.g., channel alignment and 1D-laplacian kernel fs for e-nose data; x-binning, baseline removal for TOF MS.</td>
            <td></td>
            <td>TODO</td>
        </tr>
        <tr>
            <td colspan = 2>qsi.vis</td>
            <td>Plotting</td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td colspan = 2>qsi.cs</td>
            <td>compressed sensing</td>
            <td>cs1</td>
            <td>Adaptive compressed sensing of Raman spectroscopic profiling data for discriminative tasks [J]. Talanta, 2020, doi: 10.1016/j.talanta.2019.120681
            <br/>
            Task-adaptive eigenvector-based projection (EBP) transform for compressed sensing: A case study of spectroscopic profiling sensor [J]. Analytical Science Advances. Chemistry Europe, 2021, doi: 10.1002/ansa.202100018
            <br/>
            Compressed Sensing library for spectroscopic profiling data [J]. Software Impacts, 2023, doi: 10.1016/j.simpa.2023.100492
            <br/>
            Secured telemetry based on time-variant sensing matrix – An empirical study of spectroscopic profiling, Smart Agricultural Technology, Volume 5, 2023, doi: 10.1016/j.atech.2023.100268
            </td>
        </tr>
        <tr>
            <td colspan = 1 rowspan = 3>qsi.fs</td>
        </tr>
        <tr>
            <td colspan = 1>qsi.fs.nch_time_series_fs</td>
            <td>multi-channel enose data fs with 1d-laplacian conv kernel</td>
            <td></td>
            <td>基于电子鼻和一维拉普拉斯卷积核的奶粉基粉产地鉴别</td>
        </tr>
        <tr>
            <td>qsi.fs.glasso</td>
            <td>Structured-fs of Raman data with group lasso</td>
            <td></td>
            <td>in progress</td>
        </tr>
        <tr>
            <td colspan = 2>qsi.kernel</td>
            <td>kernels</td>
            <td>ackl</td>
            <td>Analytical chemistry kernel library for spectroscopic profiling data, Food Chemistry Advances, Volume 3, 2023, 100342, ISSN 2772-753X, https://doi.org/10.1016/j.focha.2023.100342.</td>
        </tr>
        <tr>
            <td rowspan = 2>qsi.dr</td>
            <td>qsi.dr.metrics</td>
            <td>Dimensionality Reduction (DR) quality metrics</td>
            <td>pyDRMetrics, wDRMetrics</td>
            <td>pyDRMetrics - A Python toolkit for dimensionality reduction quality assessment, Heliyon, Volume 7, Issue 2, 2021, e06199, ISSN 2405-8440, doi: 10.1016/j.heliyon.2021.e06199.</td>
        </tr>
        <tr>
            <td>qsi.dr.mf</td>
            <td>matrix-factorization based DR</td>
            <td>pyMFDR</td>
            <td>Matrix Factorization Based Dimensionality Reduction Algorithms - A Comparative Study on Spectroscopic Profiling Data [J], Analytical Chemistry, 2022. doi: 10.1021/acs.analchem.2c01922</td>
        </tr>
        <tr>
            <td rowspan = 4>qsi.cla</td>
            <td>qsi.cla.metrics</td>
            <td>classifiability analysis</td>
            <td>pyCLAMs, wCLAMs</td>
            <td>A unified classifiability analysis framework based on meta-learner and its application in spectroscopic profiling data [J]. Applied Intelligence, 2021, doi: 10.1007/s10489-021-02810-8
            <br/> 
            pyCLAMs: An integrated Python toolkit for classifiability analysis [J]. SoftwareX, 2022, doi: 10.1016/j.softx.2022.101007</td>
        </tr>
        <tr>
            <td>qsi.cla.ensemble</td>
            <td>homo-stacking, hetero-stacking, FSSE</td>
            <td rowspan = 3>pyNNRW</td>
            <td rowspan = 3>Spectroscopic Profiling-based Geographic Herb Identification by Neural Network with Random Weights [J]. Spectrochimica Acta Part A: Molecular and Biomolecular Spectroscopy, 2022, doi: 10.1016/j.saa.2022.121348</td>
        </tr>
        <tr>
            <td>qsi.cla.kernel</td>
            <td>kernel-NNRW</td>
        </tr>
        <tr>
            <td>qsi.cla.nnrw</td>
            <td>neural networks with random weights</td>
        </tr>
        <tr>
            <td rowspan = 1>qsi.pipeline</td>
            <td></td>
            <td>General data analysis pipelines.</td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td rowspan = 1>qsi.gui</td>
            <td></td>
            <td colspan = 3>Web-based apps. e.g., `python -m qsi.gui.chaihu` will launch the app for bupleurum origin discrimination.</td>
        </tr>
    </tbody>
</table>
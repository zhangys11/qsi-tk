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
            <td colspan = 1 rowspan = 2>qsi.io</td>
            <td>
            <td>File I/O, Dataset loading</td>
            <td></td>
            <td>TODO qsi-tk open datasets with algorithms</td>
        </tr>
        <tr>
            <td colspan = 1>qsi.io.aug</td>
            <td>Data augmentation, e.g., generative models</td>
            <td></td>
            <td>TODO Data aug with deep generative models. e.g., " variational autoencoders, generative adversarial networks, autoregressive models, normalizing flow models, energy-based models, and score-based models. "</td>
        </tr>
        <tr>
            <td colspan = 2>qsi.vis</td>
            <td>Plotting</td>
        </tr>
        <tr>
            <td colspan = 2>qsi.cs</td>
            <td>compressed sensing</td>
            <td>cs1</td>
            <td>Adaptive compressed sensing of Raman spectroscopic profiling data for discriminative tasks [J]. Talanta, 2020, doi: 10.1016/j.talanta.2019.120681
            <br/>
            Task-adaptive eigenvector-based projection (EBP) transform for compressed sensing: A case study of spectroscopic profiling sensor [J]. Analytical Science Advances. Chemistry Europe, 2021, doi: 10.1002/ansa.202100018
            </td>
        </tr>
        <tr>
            <td colspan = 2>qsi.fs</td>
            <td>feature selection</td>
        </tr>
        <tr>
            <td colspan = 2>qsi.ks</td>
            <td>kernels</td>
            <td>ackl</td>
            <td>TODO</td>
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
        </tr>
    </tbody>
</table>
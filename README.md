# Pipeline for measuring the scintillation of fast radio bursts

This pipeline aims to automatically measure the autocorrelation function for CHIME baseband FRBs without the need for humans to interfere, e.g. identifying the FRB on-pulse range by hand.
It is written in an object-oriented manner, so that each key step in the pipeline uses a class.

The full pipeline is `FitScintPipeline.FitScintPipeline.run()`.
Given FRB ID, DM, and the output folder path, the pipeline flow works as explained below:
* `WaterfallLoader` reads in the raw baseband data, dedisperse the raw data, and mask bad channels and RFI
* `SpectraCalculator` measures the on-pulse range, upchannelizes the data in the on-pulse range, and upchannelizes random off-pulse ranges
* `SpectraFitter` fits smooth spectra for both on-pulse region and off-pulse ranges
* `ACFCalculator` measures the autocorrelation functions for the on-pulse and off-pulse regions
* `ACFFitter` fits Lorenzian to the measured autocorrelation functions

Each step above can be taken out individually for testing.

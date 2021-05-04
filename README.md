# The Augmented Theorist

This repository contains the trained models presented in the paper "The Augmented Theorist - Toward Automated Knowledge Extraction from Conceptual Models". We cloned AlexeyAB's darknet repo (https://github.com/AlexeyAB/darknet) and extended it with useful functions for the application of YOLOv4 to figure detection in scientific papers and construct, item and path coefficient detection in graphical representations of Structural Equation Models. 

## Example (on Colab)


<img  src="https://miro.medium.com/max/397/1*L2u_koKpa1lcjvB8DEDHsg.jpeg"  alt=""  width="20%"  />

[This notebook](https://github.com/purplesweatshirt/icispaper/blob/main/example.ipynb) demonstrates the workflow of our pipeline and is designed to be executed on Colab. Due to the double blind review process, we could not provide a direct link to our Colab Notebook.



## Brief overview

The following important files can be found in these directories:
- the detection_utils_new.py file, which contains all of our wrapper functions
- the cfg files are located in the cfg folder
- the data and names files are located in the data folder
- the pdf files, which are used to demonstrate the workflow

The data is obtained from [ADNI](http://adni.loni.ucla.edu) dataset, acquired by structural Magnetic Resonance Imaging (MRI) scan with both 1.5 and 3.0 Tesla MRI scan magnetic field strength. In total, the dataset contains n = 752 samples, with 126 AD, 433 Mild Cognitive Impairment (MCI)  and 193 Normal Controls (NC). For each image, we implement the Dartel VBM <sup>1</sup> for pre-processing, followed by the toolbox Statistical Parametric Mapping (SPM) for segmentation of gray matter (GM), white matter (WM) and cerebral spinal fluid (CSF). Then we use Automatic Anatomical Labeling (AAL) atlas <sup>2</sup> to partition the whole brain into  90 Cerebrum brain anatomical regions, with the volume of each region (summation of all GMs in the region) provided. 

The response variable vector is the Alzheimer's Disease Assessment Scale (ADAS), which was originally designed to assess the severity of cognitive dysfunction <sup>3</sup> and was later found to be able to clinically distinguish the diagnosed AD from normal controls <sup>4</sup>.

1. J. Ashburner. A fast diffeomorphic image registration algorithm. Neuroimage, 38(1):95–113, 2007.

2. N.  Tzourio-Mazoyer,  B. Landeau,  D. Papathanassiou,  F. Crivello,  O. Etard,  N. Delcroix, B. Mazoyer, and M. Joliot.  Automated anatomical labeling of activations in spm using amacroscopic anatomical parcellation of the MNI MRI single-subject brain. Neuroimage, 15(1):273–289, 2002.

3. W. G. Rosen, R. C. Mohs, and K. L. Davis.  A new rating scale for alzheimer’s disease. Am J Psychiatry, 141(11):1356–64, 1984.

4. R. F. Zec, E. S. Landreth, S. K. Vicari, E. Feldman, J. Belman, A. Andrise,  R. Robbs, V. Kumar, and R. Becker. Alzheimer disease assessment scale: useful for both early detection and staging of dementia of the alzheimer type. Alzheimer Disease and Associated Disorders, 1992.

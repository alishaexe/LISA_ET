# LISA_ET

## Data files
Rtab is the data file for LISA and ciao4 is the data file for ET.

## 100_iteration_table folder 
This folder contains a link to the .npy files which are the data for the Amin, Atab, Ftab values for all the BPLS curves in the instances of ET, LISA, and combines curves. The accompnying Python script
contains the code, similar to the main script for 100 iterations, but does not contain the functions for calculating the data tables. It only loads the files.

## 100 iterations script
This is the full script useing 100 values for frequencies, and nt values. The parts that calculate the BPL curves for ET, LISA, and combined will TAKE AN HOUR EACH on a powerful desktop computer I imagine laptops and low powered computers it will take much longer. This is the script that the data tables have been saved from. If this code get's updated in a way that will alter the values/change any calculations the data tables in the folder will need to be updated.

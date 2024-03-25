# LISA_ET

## Data files
Rtab is the data file for LISA and ciao4 is the data file for ET.

## 50_iteration script
This is the same as the other main scripts but with 50 values for the iterations again BPL curves take a while since 50*50*50*50 probably about 10-20 mins to run the full script in total. If it takes too long change the value of ITERA at the beginning to something like 37 and that will mean the code will run fast but you will see if the plots produced that the functions are more jagged.

## 100_iteration_table folder 
This folder contains a link to the .npy files which are the data for the Amin, Atab, Ftab values for all the BPLS curves in the instances of ET, LISA, and combines curves. The accompnying Python script
contains the code, similar to the main script for 100 iterations, but does not contain the functions for calculating the data tables. It only loads the files.

## 100 iterations script
This is the full script useing 100 values for frequencies, and nt values. The parts that calculate the BPL curves for ET, LISA, and combined will TAKE AN HOUR EACH on a powerful desktop computer I imagine laptops and low powered computers it will take much longer. This is the script that the data tables have been saved from. If this code get's updated in a way that will alter the values/change any calculations the data tables in the folder will need to be updated.


## Fisher Matrix
Fisher Matrix scripts do what they say on the tin and calc the elements of the fisher matrices and then plot the getdist plots. There is some issues with the axes scaling on these plots. For example if your \Omega_* ~ 10^10 and nt is 2/3; on the axes it'll have \Omega_* ~ 10^-7 and nt ~ 10^3. I don't know why it does this... yet. BPL is a wip as I concocted how the FM is calculated myself.

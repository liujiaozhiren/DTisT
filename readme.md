# DTisT

The details of the formulas in the paper can be found in the technical_report.pdf.
The code utilizes the `traj-dist` library for calculating distances/similarities and related matched point pairs between trajectories.

## Note

Please note that the `traj-dist`(https://github.com/bguillouet/traj-dist) library typically does not include point-pair information. 

Therefore, you may need to modify the traj-dist library to make it output the corresponding point-pair information (in the format (sim_value:float32, (u:[], v:[])) where u and v are lists of equal length, representing point pair indices in the trajectory). 

Alternatively, you can refer to the porc.data_construct.lazy_index_list_select4list method to customize the calculation of similarity for point pairs.

## Running the Code

To run the code, follow these steps:

1. Install the required dependencies by requirements.txt

2. Run the following command to execute `main.py`:

   ```bash
   python main.py
   ```

   The code will start running, using the provided sample dataset for training.

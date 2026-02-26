# Sustainable_SE
This project is part of the course [CS4575 Sustainable Software Engineering](luiscruz.github.io/course_sustainableSE/2026/) as taught at Delft University of Technology in the academic year 2025-2026.
 #### Prerequisites:
- Operating System: Linux (tested on Ubuntu 20.04+)
- [Energibridge](https://github.com/tdurieux/EnergiBridge)

### How to run the experiment:

0. Download dataset  
   - Create a folder named `dataset` inside the `experiment` folder
   - Download the COCO dataset at http://images.cocodataset.org/zips/val2017.zip
   - Unzip to `dataset` folder

1. Install requirements:
   ```
   pip install -r requirements.txt
   ```
2. Set Environment Variable (Optional)

   By default, the script looks for Energibridge at:
   ```
   $HOME/EnergiBridge/target/release/energibridge
   ```
   If you installed it elsewhere, set the environment variable ENERGIBRIDGE_PATH:
   ```
   export ENERGIBRIDGE_PATH=/path/to/EnergiBridge/target/release/energibridge
   ```
3. Run the prepare script:
   Before running the experiment, you must process the dataset and download model weights:
   ```
   cd experiment

   # Make the prepare script executable
   chmod +x prepare.sh

   # Run the prepare script
   ./prepare.sh
   ```
4. Enable [Zen mode](luiscruz.github.io/2021/10/10/scientific-guide.html)   
   There is an example script `setup.py` for automating zen mode.

5. Run the script:
   ```
   cd experiment

   # Make script executable
   chmod +x run_experiment.sh

   # Run script
   ./run_experiment.sh
   ```
6. Restore Zen Mode
